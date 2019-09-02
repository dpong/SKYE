from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Add, Subtract, Lambda, BatchNormalization, GRU
from tensorflow.keras.models import Model
#from tensorflow.keras.activations import softmax
from agent.noisydense import NoisyDense
import tensorflow.keras.backend as K

#Tensorflow 2.0 Beta

class Build_model():
    tf.keras.backend.set_floatx('float64')

    def build_model(self, state_size, neurons, action_size):
        
        state_input = Input(shape=state_size)
        norm = BatchNormalization()(state_input)

        gru1 = GRU(neurons, activation='elu',return_sequences=True)(norm)
        gru1_norm = BatchNormalization()(gru1)
        gru2 = GRU(neurons, activation='elu',return_sequences=False)(gru1_norm)
        gru2_norm = BatchNormalization()(gru2)
        
        d2 = NoisyDense(neurons, activation='elu', name='Noisy_1')(gru2_norm)
        d2_norm = BatchNormalization()(d2)

        d3_a = NoisyDense(neurons, activation='elu', name='Noisy_2')(d2_norm)
        d3_a_norm = BatchNormalization()(d3_a)
        a = NoisyDense(action_size,activation='linear', name='Noisy_3')(d3_a_norm)
        
        return Model(inputs=state_input, outputs=a)