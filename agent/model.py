from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Add, Subtract, Lambda, BatchNormalization, concatenate
from tensorflow.keras.layers import Conv1D, Flatten, MaxPooling1D, GlobalAveragePooling1D, Reshape
from tensorflow.keras.models import Model
#from agent.noisynet import NoisyDense
import tensorflow.keras.backend as K
from tensorflow.nn import softmax


class Build_model():
    tf.keras.backend.set_floatx('float64')
    def build_model(self, state_size, neurons, action_size, training):
        state_input = Input(shape=state_size, name='state_input', dtype='float64')
        norm = BatchNormalization()(state_input)  # 輸入標準化
        # 額外輸入
        self_state_input = Input(shape=(8,), name='self_state_input', dtype='float64')
        reshape_s = Reshape((8,))(self_state_input)
        s1 = Dense(8, activation='relu')(reshape_s)
        # 卷積層們，kernel_size為5天，一週的概念
        con1 = Conv1D(state_size[1], 5, padding='same', activation='relu')(norm)
        con2 = Conv1D(state_size[1], 5, padding='same', activation='relu')(con1)
        pool_max = MaxPooling1D(pool_size=5, strides=1, padding='same')(con2)
        flat = Flatten()(pool_max)
        flat_norm = BatchNormalization()(flat)
        # 外插 self_state_input
        connect = concatenate([flat_norm, s1])
        # 連結層
        n1 = Dense(neurons, activation='relu')(connect)
        n1_norm = BatchNormalization()(n1)
        # deuling advantage
        a = Dense(action_size, activation='linear')(n1_norm)
        a_mean = Lambda(lambda x: K.mean(x, axis=1, keepdims=True))(a)
        advantage = Subtract()([a, a_mean])
        # deuling value
        value = Dense(1, activation='linear')(n1_norm)
        # combine
        q = Add()([value, advantage])


        return Model(inputs=[state_input, self_state_input], outputs=q)
