from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Add, Subtract, Lambda, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import elu
from agent.noisynet import NoisyDense
import tensorflow.keras.backend as K

#Tensorflow 2.0 Beta

class Build_model():
    def build_model(self, state_size, neurons, action_size, atoms, training):
        #前面的LSTM層
        state_input = Input(shape=state_size)
        lstm1 = LSTM(neurons, activation='sigmoid',return_sequences=False)(state_input)

        #連結層
        d1 = Dense(neurons,activation='elu')(lstm1)
        d1_plus1 = Dense(neurons,activation='elu')(d1)
        d1_plus2 = Dense(neurons,activation='elu')(d1_plus1)
        d2 = Dense(neurons,activation='elu')(d1_plus2)

        noisy_distribution_list_a = []
        noisy_distribution_list_v = []
        duel_distribution_list_a = []
        duel_distribution_list_v = []
        duel_mean_list=[]
        advantage_list=[]
        q_list=[]

        for i in range(action_size):
            noisy_distribution_list_a.append(
                elu(NoisyDense(atoms, neurons, training, bias=True)(d2))
                )
            noisy_distribution_list_v.append(
                elu(NoisyDense(atoms, neurons, training, bias=True)(d2))
                )
            duel_distribution_list_a.append(
                elu(NoisyDense(atoms, atoms, training, bias=True)(noisy_distribution_list_a[i]))
                )
          
        value = elu(NoisyDense(atoms, atoms, training, bias=True)(noisy_distribution_list_v))
        value = tf.reshape(value,[1, action_size, atoms])

        print(value)

        for i in range(action_size): 
            duel_mean_list.append(
                Lambda(lambda x: K.mean(x, axis=1, keepdims=True))(duel_distribution_list_a[i])
                )
            advantage_list.append(
                Subtract()([duel_distribution_list_a[i], duel_mean_list[i]])
                )
            q_list.append(
            Add()([value, advantage_list[i]])
                )

        #最後compile
        model = Model(inputs=state_input, outputs=q_list)
        model.compile(loss='mse', optimizer=Adam(lr=0.0001))
        
        return model
    
    
