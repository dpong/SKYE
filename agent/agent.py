from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import tensorflow.keras.backend as K
import os, random, math
import numpy as np
from agent.prioritized_memory import Memory
from agent.model import Build_model
from tensorflow.nn import softmax, log_softmax
from tensorflow.compat.v1.train import get_or_create_global_step
from tensorflow.compat.v1.losses import huber_loss
from tensorflow.keras.utils import Progbar

config = tf.compat.v1.ConfigProto()
config.intra_op_parallelism_threads = 44
config.inter_op_parallelism_threads = 44
tf.compat.v1.Session(config=config)

class Agent:
	def __init__(self, ticker, state_size, neurons, m_path, is_eval=False):
		self.state_size = state_size # normalized previous days
		self.action_size = 4 
		self.neurons = neurons
		self.memory_size = 10000 #記憶長度
		self.memory = Memory(self.memory_size)
		self.epsilon = 1
		self.epsilon_min = 0.01
		self.epsilon_decay = 0.995
		self.gamma = 0.95
		self.batch_size = 64
		self.epoch_loss_avg = tf.keras.metrics.Mean()
		self.epochs = 50
		self.bar = Progbar(self.epochs)
		self.is_eval = is_eval
		self.checkpoint_path = m_path
		self.checkpoint_dir = os.path.dirname(self.checkpoint_path)
		self.check_index = self.checkpoint_path + '.index'   #checkpoint裡面的檔案多加了一個.index
		
		if is_eval==False:
			self.training = True
			self.model = self._model('  Model')
			self.target_model = self._model(' Target')
		else:
			self.training = False
			self.model = self._model('  Model')
			
		self.optimizer = tf.optimizers.Adam(learning_rate=0.001)
		
	def _model(self, model_name):
		ddqn = Build_model()
		model = ddqn.build_model(self.state_size, self.neurons, self.action_size)
		if os.path.exists(self.check_index):
			#如果已經有訓練過，就接著load權重
			print('-'*52+'{} Weights loaded!!'.format(model_name)+'-'*52)
			model.load_weights(self.checkpoint_path)
		else:
			print('-'*53+'Create new model!!'+'-'*53)
		
		return model
	
	# 把model的權重傳給target model
	def update_target_model(self):
		self.target_model.set_weights(self.model.get_weights())

	def act(self, state):
		if not self.is_eval and np.random.rand() <= self.epsilon:
			return random.randrange(self.action_size)
		options = self.model(state)
		options = options.numpy()
		return np.argmax(options[0]) # array裡面最大值的位置號

	# Prioritized experience replay
	# save sample (error,<s,a,r,s'>) to the replay memory
	def append_sample(self, state, action, reward, next_state, done):
		max_p = np.max(self.memory.tree.tree[-self.memory.capacity:])
		if max_p == 0:
			max_p = self.memory.abs_err_upper  # clipped abs error feat 莫煩
		self.memory.add(max_p, (state, action, reward, next_state, done))  # set the max p for new p

	# loss function
	def _loss(self, model, x, y):
		y_ = self.model(x)
		return huber_loss(y, y_)
	# gradient
	def _grad(self, model, inputs, targets):
		with tf.GradientTape() as tape:
			loss_value = self._loss(self.model, inputs, targets)
		return loss_value, tape.gradient(loss_value, self.model.trainable_variables)

	def train_model(self):
		# pick samples from prioritized replay memory (with batch_size)
		mini_batch, idxs, is_weights = self.memory.sample(self.batch_size)

		state_inputs = np.zeros((self.batch_size,self.state_size[0],self.state_size[1]))
		next_states = np.zeros((self.batch_size,self.state_size[0],self.state_size[1]))
		action, reward, done = [], [], []
		
		for i in range(self.batch_size):
			state_inputs[i][:][:] = mini_batch[i][0]
			action.append(mini_batch[i][1])
			reward.append(mini_batch[i][2])
			next_states[i][:][:] = mini_batch[i][3]
			done.append(mini_batch[i][4])
		
		old = []
		# 主model動作
		result = self.model(state_inputs)
		result = result.numpy()
		next_result = self.model(next_states)
		next_result = next_result.numpy()
		next_action = np.argmax(next_result, axis=1)
		# target model動作
		t_next_result = self.target_model(next_states)
		t_next_result = t_next_result.numpy()
		# 更新Q值: Double DQN的概念
		for i in range(self.batch_size):
			old.append(result[i][action[i]])
			result[i][action[i]] = reward[i]
			if not done[i]:
				result[i][action[i]] = result[i][action[i]] + self.gamma * t_next_result[i][next_action[i]]
			# 計算error給PER
			error = abs(old[i] - result[i][action[i]])
			error *= is_weights[i]
			self.memory.update(idxs[i], error)

		# train model
		for i in range(self.epochs):
			loss_value, grads = self._grad(self.model, state_inputs, result)
			self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables),
				get_or_create_global_step())
			self.epoch_loss_avg(loss_value)
			self.bar.update(i, values=[('loss', self.epoch_loss_avg.result().numpy())])

		if self.epsilon > self.epsilon_min:
			#貪婪度遞減   
			self.epsilon *= self.epsilon_decay 




	