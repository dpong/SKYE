from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import tensorflow.keras.backend as K
import os, random, math
import numpy as np
from agent.prioritized_memory import Memory
from agent.model import Build_model
from tensorflow.compat.v1.train import get_or_create_global_step
from tensorflow.keras.utils import Progbar

config = tf.compat.v1.ConfigProto()
config.intra_op_parallelism_threads = 44
config.inter_op_parallelism_threads = 44
tf.compat.v1.Session(config=config)

class Agent:
	def __init__(self, ticker, state_size, self_state_shape, neurons, m_path, is_eval=False):
		self.state_size = state_size # normalized previous days
		self.self_feat_shape = self_state_shape
		self.action_size = 4 
		self.unit_up_limit = 10
		self.unit_down_limit = 1
		self.unit_loss_weight = 0.1/self.unit_up_limit
		self.neurons = neurons
		self.memory_size = 20000 #記憶長度
		self.memory = Memory(self.memory_size)
		self.epsilon = 1
		self.epsilon_min = 0.1
		self.epsilon_decay = 0.995
		self.gamma = 0.95
		self.batch_size = 128
		self.epoch_loss_avg = tf.keras.metrics.Mean()
		self.epochs = 5
		self.bar = Progbar(self.epochs)
		self.is_eval = is_eval
		self.checkpoint_path = m_path
		self.checkpoint_dir = os.path.dirname(self.checkpoint_path)
		self.check_index = self.checkpoint_path + '.index'   #checkpoint裡面的檔案多加了一個.index
		if is_eval==False:
			self.model = self._model('  Model')
			self.target_model = self._model(' Target')
		else:
			self.model = self._model('  Model')
		self.optimizer = tf.optimizers.Adam(learning_rate=0.0001, epsilon=0.000025)
		self.loss_function = tf.keras.losses.Huber()


	def _model(self, model_name):
		ddqn = Build_model()
		model = ddqn.build_model(self.state_size, self.self_feat_shape, self.neurons, self.action_size)
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

	def act(self, state, self_state):
		if not self.is_eval and np.random.rand() <= self.epsilon:
			return random.randrange(self.action_size), random.randint(1, self.unit_up_limit)
		options, unit = self.model([state, self_state])
		options = options.numpy()
		unit = unit.numpy()
		action_out = np.argmax(options[0])  # array裡面最大值的位置號
		unit_seed = int(unit[0][0])
		if unit_seed > self.unit_up_limit:
			unit_seed = int(self.unit_up_limit)
		if unit_seed < self.unit_down_limit:
			unit_seed = int(self.unit_down_limit)
		return action_out, unit_seed

	# Prioritized experience replay
	# save sample (error,<s,a,r,s'>) to the replay memory
	def append_sample(self, state, action, reward, next_state, done):
		if reward != 0:
			max_p = 100 + reward  # 有reward的記憶給大的priority
		else:
			max_p = 10  # 預設給10
		self.memory.add(max_p, (state, action, reward, next_state, done))  # set the max p for new p

	# loss function
	def _loss(self, model, x, y):
		q_y = tf.convert_to_tensor(y[0])
		unit_y = tf.convert_to_tensor(y[1])
		q_y_, unit_y_ = self.model(x)
		q_loss = self.loss_function(y_true=q_y, y_pred=q_y_)
		unit_loss = self.loss_function(y_true=unit_y, y_pred=unit_y_)
		unit_loss = tf.multiply(unit_loss, self.unit_loss_weight)
		return tf.add(q_loss, unit_loss)

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
		action, unit, reward, done = [], [], [], []

		for i in range(self.batch_size):
			state_inputs[i] = mini_batch[i][0][0]
			self_state[i] = mini_batch[i][0][1]
			action.append(mini_batch[i][1][0])
			unit.append(mini_batch[i][1][1])
			reward.append(mini_batch[i][2])
			next_states[i][:][:] = mini_batch[i][3][0]
			done.append(mini_batch[i][4])
		old_q = []
		# 主model動作
		q_result, unit_result = self.model([state_inputs, self_state])
		q_result = q_result.numpy()
		unit_result = unit_result.numpy()
		# 下一個state
		q_next_result, unit_next_result = self.model([next_states, self_state])
		q_next_result = q_next_result.numpy()
		#unit_next_result = unit_next_result.numpy()
		next_action = np.argmax(q_next_result, axis=1)
		# target model動作
		q_t_next_result, unit_t_next_result = self.target_model([next_states, self_state])
		q_t_next_result = q_t_next_result.numpy()
		#unit_t_next_result = unit_t_next_result.numpy()
		# 更新Q值: Double DQN的概念
		for i in range(self.batch_size):
			old_q.append(q_result[i][action[i]])
			q_result[i][action[i]] = reward[i]
			if not done[i]:
				q_result[i][action[i]] = q_result[i][action[i]] + self.gamma * q_t_next_result[i][next_action[i]]
			# 計算error給PER，unit的權重打折
			error = abs(old_q[i] - q_result[i][action[i]]) + self.unit_loss_weight * abs(unit[i] - unit_result[i][0]) 
			error *= is_weights[i]
			self.memory.update(idxs[i], error)
			unit_result[i][0] = unit[i]  # 更新回array裡

		# train model
		for i in range(self.epochs):
			loss_value, grads = self._grad(self.model, [state_inputs, self_state], [q_result, unit_result])
			self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables),
				get_or_create_global_step())
			self.epoch_loss_avg(loss_value)
			self.bar.update(i, values=[('loss', self.epoch_loss_avg.result().numpy())])
		print('\n')
		if self.epsilon > self.epsilon_min:
			#貪婪度遞減   
			self.epsilon *= self.epsilon_decay 




	