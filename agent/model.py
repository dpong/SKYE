from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Add, Subtract, Lambda, BatchNormalization, GRU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import softmax
from agent.noisynet import NoisyDense
import tensorflow.keras.backend as K

#Tensorflow 2.0 Beta

class Build_model():
    def build_model(self, state_size, neurons, action_size, atoms, training):
        state_input = Input(shape=state_size)
        # 前面的GRU層
        gru1 = GRU(neurons, activation='elu',return_sequences=True)(state_input)
        gru1_norm = BatchNormalization()(gru1)
        gru2 = GRU(neurons, activation='elu',return_sequences=True)(gru1_norm)
        gru2_norm = BatchNormalization()(gru2)
        gru3 = GRU(neurons, activation='elu',return_sequences=True)(gru2_norm)
        gru3_norm = BatchNormalization()(gru3)
        gru4 = GRU(neurons, activation='elu',return_sequences=False)(gru3_norm)
        gru4_norm = BatchNormalization()(gru4)
        # 連結層
        d2 = Dense(neurons, activation='elu')(gru4_norm)
        d2_norm = BatchNormalization()(d2)
        # 開始 distribution
        noisy_distribution_list_a = []
        noisy_distribution_list_v = []
        noisy_norm_list_a = []
        noisy_norm_list_v = [] 
        duel_distribution_list_a = []
        advantage_list=[]
        output_list=[]
        # 建立一堆 Noisy 層 with elu activations
        for i in range(action_size):
            noisy_distribution_list_a.append(
                NoisyDense(atoms, neurons, activation='elu', Noisy=training, bias=False)(d2)
                )
            noisy_norm_list_a.append(
                BatchNormalization()(noisy_distribution_list_a[i])
            )
            noisy_distribution_list_v.append(
                NoisyDense(atoms, neurons, activation='elu', Noisy=training, bias=False)(d2)
                )
            noisy_norm_list_v.append(
                BatchNormalization()(noisy_distribution_list_v[i])
            )
            duel_distribution_list_a.append(
                NoisyDense(atoms, atoms, activation='linear', Noisy=training, bias=True)(noisy_norm_list_a[i])
                )
        # deuling 計算 value     
        value_in = tf.concat([t for t in noisy_norm_list_v], 1)
        value = NoisyDense(atoms, atoms * action_size, activation='linear', Noisy=training, bias=True)(value_in)
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
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001, clipnorm=0.001))
        
        return model
    
