from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Add, Subtract, Lambda, BatchNormalization, concatenate
from tensorflow.keras.layers import Conv1D, Flatten, MaxPooling1D, GlobalAveragePooling1D, Embedding
from tensorflow.keras.models import Model
#from agent.noisynet import NoisyDense
import tensorflow.keras.backend as K


class Build_model():
    tf.keras.backend.set_floatx('float64')
    def build_model(self, state_size, self_feat_shape, neurons, action_size):
        state_input = Input(shape=state_size, name='state_input', dtype='float64')
        # 額外輸入，自身的states
        self_state_input = Input(shape=(self_feat_shape[-2], self_feat_shape[-1]), name='self_state_input', dtype='float64')
        s1 = Embedding(self_feat_shape[-1], self_feat_shape[-1], input_length=self_feat_shape[-2])(self_state_input)
        flat_s1 = Flatten()(s1)
        flat_norm_s1 = BatchNormalization()(flat_s1)
        # 卷積層們，kernel_size為5天，一週的概念
        con1 = Conv1D(state_size[-1]*state_size[-1], 5, padding='causal', activation='elu')(state_input)
        con1_norm = BatchNormalization()(con1)
        con2 = Conv1D(state_size[-1]*state_size[-1], 5, padding='causal', activation='elu')(con1_norm)
        con2_norm = BatchNormalization()(con2)
        con3 = Conv1D(state_size[-1]*state_size[-1], 5, padding='causal', activation='elu')(con2_norm)
        con3_norm = BatchNormalization()(con3)
        pool_max = MaxPooling1D(pool_size=5, strides=1, padding='same')(con3_norm)
        max_norm = BatchNormalization()(pool_max)
        flat = Flatten()(max_norm)
        flat_norm = BatchNormalization()(flat)
        # 外插 self_state_input
        connect = concatenate([flat_norm, flat_norm_s1])
        # 連結層
        n1 = Dense(neurons, activation='elu')(connect)
        n1_norm = BatchNormalization()(n1)
        n2 = Dense(neurons, activation='elu')(n1_norm)
        n2_norm = BatchNormalization()(n2)
        # deuling advantage
        a = Dense(action_size, activation='linear')(n2_norm)
        a_mean = Lambda(lambda x: K.mean(x, axis=1, keepdims=True))(a)
        advantage = Subtract()([a, a_mean])
        # deuling value
        value = Dense(1, activation='linear')(n2_norm)
        # combine
        q_out = Add()([value, advantage])

        # unit network
        unit_1 = Dense(action_size, activation='elu')(n2_norm)
        unit_1_norm = BatchNormalization()(unit_1)
        unit_out = Dense(1, activation='relu')(unit_1_norm)


        return Model(inputs=[state_input, self_state_input], outputs=[q_out, unit_out])
