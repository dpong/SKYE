from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import tensorflow.keras.backend as K
import os, random, math
import numpy as np
from agent.prioritized_memory import Memory
from agent.model import Build_model
from tensorflow.nn import softmax, log_softmax

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
		self.gamma = 0.95
		self.batch_size = 128
		self.is_eval = is_eval
		self.checkpoint_path = m_path
		self.checkpoint_dir = os.path.dirname(self.checkpoint_path)
		self.check_index = self.checkpoint_path + '.index'   #checkpoint裡面的檔案多加了一個.index
		
		if is_eval==False:
			self.model = self._model('  Model', training=True)
			self.target_model = self._model(' Target', training=True)
		else:
			self.model = self._model('  Model', training=False)
		
	def _model(self, model_name, training):
		ddqn = Build_model()
		model = ddqn.build_model(self.state_size, self.neurons, self.action_size, training)
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
		options = self.model.predict(state)
		return np.argmax(options[0]) # array裡面最大值的位置號

	# Prioritized experience replay
	# save sample (error,<s,a,r,s'>) to the replay memory
	def append_sample(self, state, action, reward, next_state, done):
		max_p = np.max(self.memory.tree.tree[-self.memory.capacity:])
		if max_p == 0:
			max_p = self.memory.abs_err_upper  # clipped abs error feat 莫煩
		self.memory.add(max_p, (state, action, reward, next_state, done))  # set the max p for new p

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
		#主model動作
		result = self.model.predict(state_inputs)
		next_result = self.model.predict(next_states)
		next_action = np.argmax(next_result, axis=1)
		#target model動作
		t_next_result = self.target_model.predict(next_states)
		#更新Q值: Double DQN的概念
		for i in range(self.batch_size):
			old.append(result[i][action[i]])
			result[i][action[i]] = reward[i]
			if not done[i]:
				result[i][action[i]] = result[i][action[i]] + self.gamma * t_next_result[i][next_action[i]]
			#計算error給PER
			error = abs(old[i] - result[i][action[i]])
			error *= is_weights[i]
			self.memory.update(idxs[i], error)
			
		#train model
		self.model.fit(state_inputs, result, batch_size=self.batch_size, epochs = 1,
			 verbose=0)
	
	def clear_sess(self):
		K.clear_session()
		


	