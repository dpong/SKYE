from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Add, Subtract, Lambda, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
from agent.noisynet import NoisyDense

#Tensorflow 2.0 Beta

class Build_model():
    def build_model(self, state_size, neurons, action_size, training):
        #前面的LSTM層
        state_input = Input(shape=state_size)
        lstm1 = LSTM(neurons, activation='sigmoid',return_sequences=False)(state_input)

        #連結層
        d1 = Dense(neurons,activation='elu')(lstm1)
        d1_plus1 = Dense(neurons,activation='elu')(d1)
        d1_plus2 = Dense(neurons,activation='elu')(d1_plus1)
        d2 = Dense(neurons,activation='elu')(d1_plus2)
        
        #dueling
        d3_a = Dense(neurons/2, activation='elu')(d2)
        d3_v = Dense(neurons/2, activation='elu')(d2)
        a = Dense(action_size,activation='elu')(d3_a)
        value = Dense(1,activation='elu')(d3_v)
        a_mean = Lambda(lambda x: K.mean(x, axis=1, keepdims=True))(a)
        advantage = Subtract()([a, a_mean])
        q = Add()([value, advantage])

        # noisy & distributional
        distribution_list = []
        for i in range(action_size):
            distribution_list.append(NoisyDense(action_size, training, bias=True)(q))

        #最後compile
        model = Model(inputs=state_input, outputs=distribution_list)
        model.compile(loss='mse', optimizer=Adam(lr=0.0001))
        
        return model
    
    
