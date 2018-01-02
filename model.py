import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import legacy_seq2seq
import numpy as np


class Model():
	def __init__(self, args, training=True):
		self.args = args
		if not training:
			args.batch_size = 10 #RMM TMP, was 1
			args.seq_length = 1

		if args.model == 'rnn':
			cell_fn = rnn.BasicRNNCell
		elif args.model == 'gru':
			cell_fn = rnn.GRUCell
		elif args.model == 'lstm':
			cell_fn = rnn.BasicLSTMCell
		elif args.model == 'nas':
			cell_fn = rnn.NASCell
		else:
			raise Exception("model type not supported: {}".format(args.model))

		cells = []
		for _ in range(args.num_layers):
			cell = cell_fn(args.rnn_size)
			if training and (args.output_keep_prob < 1.0 or args.input_keep_prob < 1.0):
				cell = rnn.DropoutWrapper(cell,
										  input_keep_prob=args.input_keep_prob,
										  output_keep_prob=args.output_keep_prob)
			cells.append(cell)

		self.cell = cell = rnn.MultiRNNCell(cells, state_is_tuple=True)

		self.input_data = tf.placeholder(
			tf.int32, [args.batch_size, args.seq_length])
		self.targets = tf.placeholder(
			tf.int32, [args.batch_size, args.seq_length])
		self.initial_state = cell.zero_state(args.batch_size, tf.float32)

		with tf.variable_scope('rnnlm'):
			softmax_w = tf.get_variable("softmax_w",
										[args.rnn_size, args.vocab_size])
			softmax_b = tf.get_variable("softmax_b", [args.vocab_size])

		embedding = tf.get_variable("embedding", [args.vocab_size, args.rnn_size])
		inputs = tf.nn.embedding_lookup(embedding, self.input_data)

		# dropout beta testing: double check which one should affect next line
		if training and args.output_keep_prob:
			inputs = tf.nn.dropout(inputs, args.output_keep_prob)

		inputs = tf.split(inputs, args.seq_length, 1)
		inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

		def loop(prev, _):
			prev = tf.matmul(prev, softmax_w) + softmax_b
			prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
			return tf.nn.embedding_lookup(embedding, prev_symbol)

		outputs, last_state = legacy_seq2seq.rnn_decoder(inputs, self.initial_state, cell, loop_function=loop if not training else None, scope='rnnlm')
		output = tf.reshape(tf.concat(outputs, 1), [-1, args.rnn_size])


		self.logits = tf.matmul(output, softmax_w) + softmax_b
		self.probs = tf.nn.softmax(self.logits)
		loss = legacy_seq2seq.sequence_loss_by_example(
				[self.logits],
				[tf.reshape(self.targets, [-1])],
				[tf.ones([args.batch_size * args.seq_length])])
		with tf.name_scope('cost'):
			self.cost = tf.reduce_sum(loss) / args.batch_size / args.seq_length
		self.final_state = last_state
		self.lr = tf.Variable(0.0, trainable=False)
		tvars = tf.trainable_variables()
		grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
				args.grad_clip)
		with tf.name_scope('optimizer'):
			optimizer = tf.train.AdamOptimizer(self.lr)
		self.train_op = optimizer.apply_gradients(zip(grads, tvars))

		# instrument tensorboard
		tf.summary.histogram('logits', self.logits)
		tf.summary.histogram('loss', loss)
		tf.summary.scalar('train_loss', self.cost)
		
		
	def sample_montecarlo(self, sess, chars, vocab, num=200, prime='|', sampling_type=1,temperature=1.0,depth=50,width=10):
		state = sess.run(self.cell.zero_state(width, tf.float32)) 
		
		#Prime the network with the starting text.
		for char in prime[:-1]:
			x = np.zeros((width, 1))
			x.fill(vocab[char])
			feed = {self.input_data: x, self.initial_state: state}
			[state] = sess.run([self.final_state], feed)
			
		def weighted_pick(weights):
			#pick = tf.multinomial(weights,1).eval()
			#return pick
			picks = []
			for run in weights:
				t = np.cumsum(run)
				s = np.sum(run)
				pck = int(np.searchsorted(t, np.random.rand(1)*s))
				picks.append([pck])
			picks = np.array(picks)
			return picks
			
		#def softmax(x):
		#	"""Compute softmax values for each sets of scores in x."""
		#	x_e = np.exp(x)
		#	return x_e / np.sum(x_e, axis=0)
		
		def softmax(z):
			if len(z.shape) == 2:
				s = np.max(z, axis=1)
				s = s[:, np.newaxis] # necessary step to do broadcasting
				e_x = np.exp(z - s)
				div = np.sum(e_x, axis=1)
				div = div[:, np.newaxis] # dito
				return e_x / div
			else:
				x_e = np.exp(x)
				return x_e / np.sum(x_e, axis=0)

		#This sets the current state to be the state of the best performer in the batch.
		def cloneBestState(currentState,bestIndex):
			nState = []
			for layer in currentState:
				c_best = layer.c[bestIndex]
				c = np.tile(c_best, (layer.c.shape[0], 1))
				h_best = layer.h[bestIndex]
				h = np.tile(h_best, (layer.h.shape[0], 1))
				nState.append(rnn.LSTMStateTuple(c,h))
			return nState
			
		#char = np.zeros((width,1, 1))
		#char.fill(vocab[prime[-1]]) #fix, not quite
		char = vocab[prime[-1]]
		for n in range(num):
			running_probs = np.ones((width, 1))
			nextchars = None
			nextStates = None
			x = np.zeros((width, 1))
			x.fill(char)
			for d in range(depth):
				feed = {self.input_data: x, self.initial_state: state}
				[probs,logits,state] = sess.run([self.probs, self.logits,self.final_state], feed)
				p = logits
				p = p / temperature #scale by temperature
				p = softmax(p)
				pick = weighted_pick(p)
				pickprobs = np.zeros((width,1))
				for i in range(width):
					pickprobs[i][0] = p[i][pick[i][0]]
				running_probs *= pickprobs
				if nextchars is None:
					nextchars = pick
				if nextStates is None:
					nextStates = list(state)
				x = pick
			bestIndex = np.argmin(running_probs)
			#print("best: ",running_probs[bestIndex][0])
			char = nextchars[bestIndex][0] #vocab[nextchars[bestIndex][0]]
			print(chars[char],end='')
			state = cloneBestState(nextStates,bestIndex) 
			
			
			
				

	def sample(self, sess, chars, vocab, num=200, prime='|', sampling_type=1,temperature=1.0):
		state = sess.run(self.cell.zero_state(1, tf.float32))
		
		for char in prime[:-1]:
			x = np.zeros((1, 1))
			x[0, 0] = vocab[char]
			feed = {self.input_data: x, self.initial_state: state}
			[state] = sess.run([self.final_state], feed)

		def weighted_pick(weights):
			t = np.cumsum(weights)
			s = np.sum(weights)
			return(int(np.searchsorted(t, np.random.rand(1)*s)))

		ret = prime
		char = prime[-1]
		for n in range(num):
			x = np.zeros((1, 1))
			x[0, 0] = vocab[char]
			feed = {self.input_data: x, self.initial_state: state}
			[probs,logits,state] = sess.run([self.probs, self.logits,self.final_state], feed)
			p = logits[0]

			p = p / temperature #scale by temperature
			def softmax(x):
				"""Compute softmax values for each sets of scores in x."""
				return np.exp(x) / np.sum(np.exp(x), axis=0)
			p = softmax(p)
			
			if sampling_type == 0:
				sample = np.argmax(p)
			elif sampling_type == 2:
				if char == '\n':
					sample = weighted_pick(p)
				else:
					sample = np.argmax(p)
			else:  # sampling_type == 1 default:
				sample = weighted_pick(p)

			pred = chars[sample]
			print(pred,end='')
			ret += pred
			char = pred
		#return ret