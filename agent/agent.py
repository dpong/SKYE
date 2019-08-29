from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import os, random, math
import numpy as np
from agent.prioritized_memory import Memory
from agent.model import Build_model
from tensorflow.nn import softmax, log_softmax

config = tf.compat.v1.ConfigProto()
config.intra_op_parallelism_threads = 8
config.inter_op_parallelism_threads = 8
tf.compat.v1.Session(config=config)

class Agent:
	def __init__(self, ticker, state_size, neurons, m_path, is_eval=False):
		self.state_size = state_size # normalized previous days
		self.action_size = 4 
		self.neurons = neurons
		self.memory_size = 1000 #記憶長度
		self.memory = Memory(self.memory_size)
		self.gamma = 0.95
		self.batch_size = 32

		self.num_atoms = 51 # for C51
		self.v_max = 10 
		self.v_min = -10 
		self.delta_z = (self.v_max - self.v_min) / float(self.num_atoms - 1)
		self.z = [self.v_min + i * self.delta_z for i in range(self.num_atoms)]

		self.is_eval = is_eval
		self.checkpoint_path = m_path
		self.checkpoint_dir = os.path.dirname(self.checkpoint_path)
		self.check_index = self.checkpoint_path + '.index'   #checkpoint裡面的檔案多加了一個.index
		
		if is_eval==False:
			self.model = self._model('  Model', training=True)
			self.target_model = self._model(' Target', training=True)
		else:
			self.model = self._model('  Model', training=False)
		
		self.cp_callback = self._check_point()
		
		
	def _model(self, model_name, training):
		ddqn = Build_model()
		model = ddqn.build_model(self.state_size, self.neurons, self.action_size, self.num_atoms, training)
		if os.path.exists(self.check_index):
			#如果已經有訓練過，就接著load權重
			print('-'*52+'{} Weights loaded!!'.format(model_name)+'-'*52)
			model.load_weights(self.checkpoint_path)
		else:
			print('-'*53+'Create new model!!'+'-'*53)
		return model

	# 設定check point
	def _check_point(self):
		cp_callback = tf.keras.callbacks.ModelCheckpoint(
		filepath=self.checkpoint_path,
		save_weights_only=True,
		verbose=0)
		return cp_callback

	# 把model的權重傳給target model
	def update_target_model(self):
		self.target_model.set_weights(self.model.get_weights())

	def act(self, state):
		# distributional
		p = self.model.predict(state) # Return a list [1x51...]
		p_concat = np.vstack(p)
		q = np.sum(np.multiply(p_concat, np.array(self.z)), axis=1) 

        # Pick action with the biggest Q value
		action_idx = np.argmax(q)
		return action_idx

	# Prioritized experience replay
	# save sample (error,<s,a,r,s'>) to the replay memory
	def append_sample(self, state, action, reward, next_state, done):
		target, error = self.get_target_n_error_51(state, action, reward, next_state, done)
		self.memory.add(error,(state, action, reward, next_state, done))

	def train_model(self):
		# pick samples from prioritized replay memory (with batch_size)
		mini_batch, idxs, is_weights = self.memory.sample(self.batch_size)

		for i in range(self.batch_size):
			target, error = self.get_target_n_error_51(
				mini_batch[i][0], #state
				mini_batch[i][1], #action
				mini_batch[i][2], #reward
				mini_batch[i][3], #next_state
				mini_batch[i][4]  #done
				)

			idx = idxs[i] # update priority
			self.memory.update(idx, error)
			#train model
			self.model.fit(mini_batch[i][0], target, epochs = 1,
			 verbose=0, callbacks = [self.cp_callback])
	
	def get_target_n_error_51(self, state, action, reward, next_state, done):
		p = self.model.predict(state)
		# 一樣有 double dqn
		p_next = self.model.predict(next_state)
		p_t_next = self.target_model.predict(next_state)
		p_concat = np.vstack(p_next)
		q = np.sum(np.multiply(p_concat, np.array(self.z)), axis=1) 
		next_action_idxs = np.argmax(q)
		# init m 值
		m_prob = [np.zeros((1, self.num_atoms))]
		# action 後更新 m 值
		if done: # Distribution collapses to a single point
			Tz = min(self.v_max, max(self.v_min, reward))
			bj = (Tz - self.v_min) / self.delta_z 
			m_l, m_u = math.floor(bj), math.ceil(bj)
			m_prob[0][0][int(m_l)] += (m_u - bj)
			m_prob[0][0][int(m_u)] += (bj - m_l)
		else:
			for j in range(self.num_atoms):
				Tz = min(self.v_max, max(self.v_min, reward + self.gamma * self.z[j]))
				bj = (Tz - self.v_min) / self.delta_z
				m_l, m_u = math.floor(bj), math.ceil(bj)
				m_prob[0][0][int(m_l)] += p_t_next[next_action_idxs][0][j] * (m_u - bj)
				m_prob[0][0][int(m_u)] += p_t_next[next_action_idxs][0][j] * (bj - m_l)
		# 更新後放回p，回去訓練
		p[action][0][:] = m_prob[0][0][:]
		# 計算 cross entropy loss
		error = -tf.reduce_sum(m_prob[0] * tf.math.log(p[action]))
		
		return p, error



	