from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import tensorflow.keras.backend as K
import os, random, math
import numpy as np
from agent.prioritized_memory import Memory
from agent.model import Build_model
from tensorflow.compat.v1.train import get_or_create_global_step
from tensorflow.keras.utils import Progbar
from tensorflow.nn import softmax, softmax_cross_entropy_with_logits

config = tf.compat.v1.ConfigProto()
config.intra_op_parallelism_threads = 44
config.inter_op_parallelism_threads = 44
tf.compat.v1.Session(config=config)

class Agent:
	def __init__(self, ticker, state_size, neurons, m_path, is_eval=False):
		self.state_size = state_size 
		self.action_size = 3
		self.neurons = neurons
		self.memory_size = 10000 #記憶長度
		self.memory = Memory(self.memory_size)
		self.epsilon = 0.3
		self.epsilon_min = 0.05
		self.epsilon_decay = 0.995
		self.gamma = 0.95
		self.batch_size = 128
		self.num_atoms = 51 # for C51
		self.v_max = 1
		self.v_min = -1 
		self.delta_z = (self.v_max - self.v_min) / float(self.num_atoms - 1)
		self.z = [self.v_min + i * self.delta_z for i in range(self.num_atoms)]
		self.epoch_loss_avg = tf.keras.metrics.Mean()
		self.epochs = 1
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
			
		self.optimizer = tf.optimizers.Adam(learning_rate=0.0000625, epsilon = 0.00015)


	def _model(self, model_name):
		ddqn = Build_model()
		model = ddqn.build_model(self.state_size, self.neurons, self.action_size, self.num_atoms, self.training)
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
		p = self._tensor_to_np(self.model(state))
		p_concat = np.vstack(p)
		q = np.sum(np.multiply(p_concat, np.array(self.z)), axis=1)
		action_idx = np.argmax(q)
		return action_idx

	# Prioritized experience replay
	# save sample (error,<s,a,r,s'>) to the replay memory
	def append_sample(self, state, action, reward, next_state, done):
		if reward != 0:
			max_p = 100 + reward  # 有reward的記憶給大的priority
		else:
			max_p = 10  # 預設給10
		self.memory.add(max_p, (state, action, reward, next_state, done))  # set the max p for new p

	def _tensor_to_np(self, x):
		# a list of tensor [128x51,,,]
		y = []
		for i in range(self.action_size):
				x[i] = softmax(x[i])
				y.append(x[i].numpy())
		return y

	# loss function
	def _loss(self, model, x, y):
		y_ = self.model(x)
		return tf.reduce_mean(softmax_cross_entropy_with_logits(labels=y, logits=y_))
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
		m_prob = [np.zeros((self.batch_size, self.num_atoms)) for i in range(self.action_size)]
		action, reward, done = [], [], []

		for i in range(self.batch_size):
			state_inputs[i][:][:] = mini_batch[i][0][0]
			action.append(mini_batch[i][1])
			reward.append(mini_batch[i][2])
			next_states[i][:][:] = mini_batch[i][3][0]
			done.append(mini_batch[i][4])

		p = self._tensor_to_np(self.model(state_inputs))
		new_p = p
		p_next = self._tensor_to_np(self.model(next_states)) 
		p_t_next = self._tensor_to_np(self.target_model(next_states))
			
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
			
			#new_p[action[i]][i][:] = m_prob[action[i]][i][:]	
		for i in range(self.batch_size):
			error = abs(tf.reduce_sum(m_prob[action[i]][i] * np.log(p[action[i]][i]+1e-9)))
			error *= is_weights[i]
			self.memory.update(idxs[i], error)

		# train model
		for i in range(self.epochs):
			loss_value, grads = self._grad(self.model, state_inputs, m_prob)
			self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables),
				get_or_create_global_step())
			self.epoch_loss_avg(loss_value)
			
		




	