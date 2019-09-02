from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Add, Subtract, Lambda, BatchNormalization, Conv1D, Flatten, MaxPooling1D, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import softmax
from agent.noisydense import NoisyDense
import tensorflow.keras.backend as K

#Tensorflow 2.0 Beta

class Build_model():
    tf.keras.backend.set_floatx('float64')

    def build_model(self, state_size, neurons, action_size):
        
        state_input = Input(shape=state_size)
        
        con1 = Conv1D(neurons, state_size[1], padding="causal", activation='relu')(state_input)
        con_norm1 = BatchNormalization()(con1)
        con2 = Conv1D(neurons, state_size[1], padding="causal", activation='relu')(con_norm1)
        con_norm2 = BatchNormalization()(con2)
        pool_max = MaxPooling1D(pool_size=2)(con_norm2)
        flat = Flatten()(pool_max)
        
        # 連結層
        d2 = NoisyDense(neurons, activation='elu', name='Noisy_1')(flat)
        d2_norm = BatchNormalization()(d2)
        #dueling
        d3_a = NoisyDense(neurons, activation='elu', name='Noisy_2')(d2_norm)
        d3_a_norm = BatchNormalization()(d3_a)
        a = NoisyDense(action_size,activation='linear', name='Noisy_3')(d3_a_norm)
        
        return Model(inputs=state_input, outputs=a)
    
        '''
        gru1 = GRU(neurons, activation='elu',return_sequences=True)(state_input)
        gru1_norm = BatchNormalization()(gru1)
        gru2 = GRU(neurons, activation='elu',return_sequences=True)(gru1_norm)
        gru2_norm = BatchNormalization()(gru2)
        gru3 = GRU(neurons, activation='elu',return_sequences=True)(gru2_norm)
        gru3_norm = BatchNormalization()(gru3)
        gru4 = GRU(neurons, activation='elu',return_sequences=False)(gru3_norm)
        gru4_norm = BatchNormalization()(gru4)
        '''