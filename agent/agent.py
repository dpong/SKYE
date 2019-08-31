from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import os, random, math
import numpy as np
from agent.prioritized_memory import Memory
from agent.model import Build_model
from tensorflow.nn import softmax, log_softmax

config = tf.compat.v1.ConfigProto()
config.intra_op_parallelism_threads = 44
config.inter_op_parallelism_threads = 44
sess = tf.compat.v1.Session(config=config)

class Agent:
	def __init__(self, ticker, state_size, neurons, m_path, is_eval=False):
		self.state_size = state_size # normalized previous days
		self.action_size = 4 
		self.neurons = neurons
		self.memory_size = 50000 #記憶長度
		self.memory = Memory(self.memory_size)
		self.gamma = 0.95
		self.batch_size = 128

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
		
		#self.cp_callback = self._check_point()
		
		
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
	'''
	# 設定check point
	def _check_point(self):
		cp_callback = tf.keras.callbacks.ModelCheckpoint(
		filepath=self.checkpoint_path,
		save_weights_only=True,
		verbose=0)
		return cp_callback
	'''

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
		max_p = np.max(self.memory.tree.tree[-self.memory.capacity:])
		if max_p == 0:
			max_p = self.memory.abs_err_upper  # clipped abs error feat 莫煩
		self.memory.add(max_p, (state, action, reward, next_state, done))  # set the max p for new p

	def train_model(self):
		# pick samples from prioritized replay memory (with batch_size)
		mini_batch, idxs, is_weights = self.memory.sample(self.batch_size)

		state_inputs = np.zeros(((self.batch_size,) + self.state_size)) 
		next_states = np.zeros(((self.batch_size,) + self.state_size))
		m_prob = [np.zeros((self.batch_size, self.num_atoms)) for i in range(self.action_size)]
		action, reward, done = [], [], []
		
		for i in range(self.batch_size):
			state_inputs[i,:,:] = mini_batch[i][0]
			action.append(mini_batch[i][1])
			reward.append(mini_batch[i][2])
			next_states[i,:,:] = mini_batch[i][3]
			done.append(mini_batch[i][4])
		
		p = self.model.predict(state_inputs)
		p_next = self.model.predict(next_states) # Return a list [32x51, 32x51, 32x51]
		p_t_next = self.target_model.predict(next_states) # Return a list [32x51, 32x51, 32x51]
		old_q = np.sum(np.multiply(np.vstack(p), np.array(self.z)), axis=1) 
		old_q = old_q.reshape((self.batch_size, self.action_size), order='F')
		optimal_action_idxs = []
		q = np.sum(np.multiply(np.vstack(p_next), np.array(self.z)), axis=1) # length (num_atoms x num_actions)
		q = q.reshape((self.batch_size, self.action_size), order='F')
		optimal_action_idxs = np.argmax(q, axis=1)
		
		for i in range(self.batch_size):
			if done[i]: # Terminal State
				Tz = min(self.v_max, max(self.v_min, reward[i]))
				bj = (Tz - self.v_min) / self.delta_z 
				m_l, m_u = math.floor(bj), math.ceil(bj)
				m_prob[action[i]][i][int(m_l)] += (m_u - bj)
				m_prob[action[i]][i][int(m_u)] += (bj - m_l)
			else:
				for j in range(self.num_atoms):
					Tz = min(self.v_max, max(self.v_min, reward[i] + self.gamma * self.z[j]))
					bj = (Tz - self.v_min) / self.delta_z
					m_l, m_u = math.floor(bj), math.ceil(bj)
					m_prob[action[i]][i][int(m_l)] += p_t_next[optimal_action_idxs[i]][i][j] * (m_u - bj)
					m_prob[action[i]][i][int(m_u)] += p_t_next[optimal_action_idxs[i]][i][j] * (bj - m_l)
			
			p[action[i]][i][:] = m_prob[action[i]][i][:]
			
		new_q = np.sum(np.multiply(np.vstack(p), np.array(self.z)), axis=1) 
		new_q = new_q.reshape((self.batch_size, self.action_size), order='F')
		for i in range(self.batch_size):
			error = abs(old_q[i][action[i]] - new_q[i][action[i]])
			error *= is_weights[i]
			self.memory.update(idxs[i], error)
		
		#train model
		self.model.fit(state_inputs, p, batch_size=self.batch_size, epochs = 1,
			 verbose=0)

	