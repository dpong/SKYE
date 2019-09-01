from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Add, Subtract, Lambda, BatchNormalization, Conv1D, Flatten, MaxPooling1D, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import softmax
#from agent.noisynet import NoisyDense
import tensorflow.keras.backend as K

#Tensorflow 2.0 Beta

class Build_model():
    def build_model(self, state_size, neurons, action_size, training):
        # 前面的卷積層
        state_input = Input(shape=state_size)
        con1 = Conv1D(neurons, state_size[1], padding="causal", activation='relu')(state_input)
        con_norm1 = BatchNormalization()(con1)
        con2 = Conv1D(neurons, state_size[1], padding="causal", activation='relu')(con1)
        con_norm2 = BatchNormalization()(con2)
        pool_max = MaxPooling1D(pool_size=2)(con_norm2)
        con3 = Conv1D(neurons, state_size[1], padding="causal", activation='relu')(pool_max)
        con_norm3 = BatchNormalization()(con3)
        con4 = Conv1D(neurons, state_size[1], padding="causal", activation='relu')(con_norm3)
        con_norm4 = BatchNormalization()(con4)
        pool_avg = GlobalAveragePooling1D()(con_norm4)
        flat = Flatten()(pool_avg)

        # 連結層
        d1 = Dense(neurons,activation='elu')(flat)
        d1_norm = BatchNormalization()(d1)
        d2 = Dense(neurons,activation='elu')(d1_norm)
        d2_norm = BatchNormalization()(d2)
        
        #dueling
        d3_a = Dense(neurons, activation='elu')(d2_norm)
        d3_a_norm = BatchNormalization()(d3_a)
        d3_v = Dense(neurons, activation='elu')(d2_norm)
        d3_v_norm = BatchNormalization()(d3_v)
        a = Dense(action_size, activation='linear')(d3_a_norm)
        a_norm = BatchNormalization()(a)
        value = Dense(1, activation='linear')(d3_v)
        value_norm = BatchNormalization()(value)
        a_mean = Lambda(lambda x: K.mean(x, axis=1, keepdims=True))(a_norm)
        advantage = Subtract()([a_norm, a_mean])
        q = Add()([value, advantage])

        # 最後compile
        model = Model(inputs=state_input, outputs=q)
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001, clipnorm=0.001))
        
        return model
    
