from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Add, Subtract, Lambda, BatchNormalization, Conv1D, Flatten, MaxPooling1D, GlobalAveragePooling1D
from tensorflow.keras.models import Model
#from tensorflow.keras.activations import softmax
#from agent.noisydense import NoisyDense
import tensorflow.keras.backend as K

#Tensorflow 2.0 Beta

class Build_model():
    tf.keras.backend.set_floatx('float64')

    def build_model(self, state_size, neurons, action_size):
        
        state_input = Input(shape=state_size)
        norm = BatchNormalization()(state_input)

        con1 = Conv1D(neurons, state_size[1], padding="causal", activation='relu')(norm)
        con_norm1 = BatchNormalization()(con1)
        con2 = Conv1D(neurons, state_size[1], padding="causal", activation='relu')(con_norm1)
        con_norm2 = BatchNormalization()(con2)
        pool_max = MaxPooling1D(pool_size=2)(con_norm2)
        max_norm = BatchNormalization()(pool_max)
        con3 = Conv1D(neurons, state_size[1], padding="causal", activation='relu')(max_norm)
        con_norm3 = BatchNormalization()(con3)
        con4 = Conv1D(neurons, state_size[1], padding="causal", activation='relu')(con_norm3)
        con_norm4 = BatchNormalization()(con4)
        pool_avg = GlobalAveragePooling1D()(con_norm4)
        avg_norm = BatchNormalization()(pool_avg)
        flat = Flatten()(pool_avg)
        flat_norm = BatchNormalization()(flat)
        
        d2 = Dense(neurons, activation='elu')(flat_norm)
        d2_norm = BatchNormalization()(d2)
        d3_a = Dense(neurons, activation='elu')(d2_norm)
        d3_a_norm = BatchNormalization()(d3_a)
        a = Dense(action_size,activation='linear')(d3_a_norm)
        
        return Model(inputs=state_input, outputs=a)