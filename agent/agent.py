from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import tensorflow.keras.backend as K
import os, random, math
import numpy as np
from agent.prioritized_memory import Memory
from agent.model import Build_model
from tensorflow.compat.v1.train import get_or_create_global_step
from tensorflow.compat.v1.losses import huber_loss
from tensorflow.keras.utils import Progbar
from tensorflow.nn import softmax, softmax_cross_entropy_with_logits

config = tf.compat.v1.ConfigProto()
config.intra_op_parallelism_threads = 44
config.inter_op_parallelism_threads = 44
tf.compat.v1.Session(config=config)

class Agent:
	def __init__(self, ticker, state_size, self_state_shape, neurons, m_path, is_eval=False):
		self.state_size = state_size 
		self.self_feat_shape = self_state_shape
		self.action_size = 4 
		self.neurons = neurons
		self.memory_size = 20000 #記憶長度
		self.memory = Memory(self.memory_size)
		self.epsilon = 1
		self.epsilon_min = 0.01
		self.epsilon_decay = 0.999
		self.gamma = 0.95
		self.batch_size = 64
		self.num_atoms = 51 # for C51
		self.v_max = 5
		self.v_min = -5 
		self.delta_z = (self.v_max - self.v_min) / float(self.num_atoms - 1)
		self.z = [self.v_min + i * self.delta_z for i in range(self.num_atoms)]
		self.epoch_loss_avg = tf.keras.metrics.Mean()
		self.epochs = 5
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
			
		self.optimizer = tf.optimizers.Adam(learning_rate=0.00025, epsilon = 0.0003125)


	def _model(self, model_name):
		ddqn = Build_model()
		model = ddqn.build_model(self.state_size, self.self_feat_shape, self.neurons, self.action_size, self.num_atoms)
		if os.path.exists(self.check_index):
			#如果已經有訓練過，就接著load權重
			print('-'*52+'{} Weights loaded!!'.format(model_name)+'-'*52)
			model.load_weights(self.checkpoint_path)
		else:
			print('-'*53+'Create new model!!'+'-'*53)
		
		if self.is_eval == True:
			model.get_layer('n1').remove_noise()
			for i in range(self.action_size):
				model.get_layer('a_{}'.format(i)).remove_noise()
			model.get_layer('value').remove_noise()
		else:
			model.get_layer('n1').sample_noise()
			for i in range(self.action_size):
				model.get_layer('a_{}'.format(i)).sample_noise()
			model.get_layer('value').sample_noise()
			
		return model
	
	# 把model的權重傳給target model
	def update_target_model(self):
		self.target_model.set_weights(self.model.get_weights())

	def act(self, state, self_state):
		if not self.is_eval and np.random.rand() <= self.epsilon:
			return random.randrange(self.action_size)
		p = self._tensor_to_np(self.model([state, self_state]))
		p_concat = np.vstack(p)
		q = np.sum(np.multiply(p_concat, np.array(self.z)), axis=1) 
		action_idx = np.argmax(q)
		return action_idx

	# Prioritized experience replay
	# save sample (error,<s,a,r,s'>) to the replay memory
	def append_sample(self, state, action, reward, next_state, done):
		if not reward == 0:
			max_p = 1000  #如果有動靜則給超大的
		else:
			max_p = 1  # 預設給1
		#max_p = np.max(self.memory.tree.tree[-self.memory.capacity:])
		#if max_p == 0:
		#	max_p = self.memory.abs_err_upper
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
		self_state = np.zeros((self.batch_size, self.self_feat_shape[-2], self.self_feat_shape[-1]))
		m_prob = [np.zeros((self.batch_size, self.num_atoms)) for i in range(self.action_size)]
		action, reward, done = [], [], []
		
		for i in range(self.batch_size):
			state_inputs[i][:][:] = mini_batch[i][0][0]
			self_state[i] = mini_batch[i][0][1]
			action.append(mini_batch[i][1])
			reward.append(mini_batch[i][2])
			next_states[i][:][:] = mini_batch[i][3][0]
			done.append(mini_batch[i][4])

		p = self._tensor_to_np(self.model([state_inputs, self_state]))
		#new_p = p
		p_next = self._tensor_to_np(self.model([next_states, self_state])) 
		p_t_next = self._tensor_to_np(self.target_model([next_states, self_state]))
			
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
			loss_value, grads = self._grad(self.model, [state_inputs, self_state], m_prob)
			self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables),
				get_or_create_global_step())
			self.epoch_loss_avg(loss_value)
			self.bar.update(i, values=[('loss', self.epoch_loss_avg.result().numpy())])
		print('\n')
		if self.epsilon > self.epsilon_min:
			#貪婪度遞減   
			self.epsilon *= self.epsilon_decay 




	