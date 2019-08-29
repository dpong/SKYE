from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Add, Subtract, Lambda, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import softmax
from agent.noisynet import NoisyDense
import tensorflow.keras.backend as K

#Tensorflow 2.0 Beta

class Build_model():
    def build_model(self, state_size, neurons, action_size, atoms, training):
        # 前面的LSTM層
        state_input = Input(shape=state_size)
        lstm1 = LSTM(neurons, activation='sigmoid',return_sequences=False)(state_input)

        # 連結層
        d1 = Dense(neurons,activation='elu')(lstm1)
        d1_plus1 = Dense(neurons,activation='elu')(d1)
        d1_plus2 = Dense(neurons,activation='elu')(d1_plus1)
        d2 = Dense(neurons,activation='elu')(d1_plus2)
        
        # 開始 distribution
        noisy_distribution_list_a = []
        noisy_distribution_list_v = []
        duel_distribution_list_a = []
        duel_distribution_list_v = []
        duel_mean_list=[]
        advantage_list=[]
        output_list=[]
        # 建立一堆 Noisy 層 with elu activations
        for i in range(action_size):
            noisy_distribution_list_a.append(
                NoisyDense(atoms, neurons, activation='elu', Noisy=training, bias=True)(d2)
                )
            noisy_distribution_list_v.append(
                NoisyDense(atoms, neurons, activation='elu', Noisy=training, bias=True)(d2)
                )
            duel_distribution_list_a.append(
                NoisyDense(atoms, atoms, activation='elu', Noisy=training, bias=True)(noisy_distribution_list_a[i])
                )
        
        # deuling 計算 value     
        value_in = tf.concat([t for t in noisy_distribution_list_v], 1)
        value = NoisyDense(atoms, atoms * action_size, activation='elu', Noisy=training, bias=True)(value_in)
        # Output shape is (None, atoms) 

        # dueling 計算 a 的 mean 值
        a_mean = tf.reduce_mean(duel_distribution_list_a, 0)
        
        # (a - a_mean) + value 
        for i in range(action_size): 
            advantage_list.append(
                Subtract()([duel_distribution_list_a[i], a_mean])
                )
            output_list.append(
            softmax(Add()([value, advantage_list[i]]))
                )

        # 最後compile
        model = Model(inputs=state_input, outputs=output_list)
        model.compile(loss='mse', optimizer=Adam(lr=0.001))
        
        return model
        # 輸出的值用什麼 activation 會再外面決定
    
