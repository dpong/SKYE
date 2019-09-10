from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Add, Subtract, Lambda, BatchNormalization, concatenate, Activation
from tensorflow.keras.layers import Conv1D, Flatten, MaxPooling1D, GlobalAveragePooling1D, Embedding, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
#from agent.noisynet import NoisyDense
import tensorflow.keras.backend as K


class Build_model():
    tf.keras.backend.set_floatx('float64')
    def build_model(self, state_size, self_feat_shape, neurons, action_size):
        state_input = Input(shape=state_size, name='state_input', dtype='float64')
        drop0 = Dropout(0.2)(state_input)
        # 額外輸入，自身的states
        self_state_input = Input(shape=(self_feat_shape[-2], self_feat_shape[-1]), name='self_state_input', dtype='float64')
        s1 = Embedding(self_feat_shape[-1], self_feat_shape[-1], input_length=self_feat_shape[-2])(self_state_input)
        flat_s1 = Flatten()(s1)
        # 卷積層們，kernel_size為5天，一週的概念
        con1 = Conv1D(state_size[-1]*state_size[-1], 5, padding='causal')(drop0)
        con1_norm = BatchNormalization()(con1)
        con1_norm_act = Activation('elu')(con1_norm)
        con2 = Conv1D(state_size[-1]*state_size[-1], 5, padding='causal')(con1_norm_act)
        con2_norm = BatchNormalization()(con2)
        con2_norm_act = Activation('elu')(con2_norm)
        pool_max = MaxPooling1D(pool_size=5, strides=1, padding='same')(con2_norm_act)
        con3 = Conv1D(state_size[-1]*state_size[-1], 5, padding='causal')(pool_max)
        con3_norm = BatchNormalization()(con3)
        con3_norm_act = Activation('elu')(con3_norm)
        con4 = Conv1D(state_size[-1]*state_size[-1], 5, padding='causal')(con3_norm_act)
        con4_norm = BatchNormalization()(con4)
        con4_norm_act = Activation('elu')(con4_norm)
        pool_max_2 = MaxPooling1D(pool_size=5, strides=1, padding='same')(con4_norm_act)
        flat = Flatten()(pool_max_2)
        drop1 = Dropout(0.3)(flat)
        # 外插 self_state_input
        connect = concatenate([drop1, flat_s1])
        # 連結層
        n1 = Dense(neurons, kernel_regularizer=l2(0.01))(connect)
        n1_norm = BatchNormalization()(n1)
        n1_norm_act = Activation('elu')(n1_norm)
        drop2 = Dropout(0.5)(n1_norm_act)
        n2 = Dense(neurons, kernel_regularizer=l2(0.01))(drop2)
        n2_norm = BatchNormalization()(n2)
        n2_norm_act = Activation('elu')(n2_norm)
        drop3 = Dropout(0.5)(n2_norm_act)
        # deuling advantage
        n3_a = Dense(neurons/2)(drop3)
        n3_a_norm = BatchNormalization()(n3_a)
        n3_a_norm_act = Activation('elu')(n3_a_norm)
        drop4_a = Dropout(0.3)(n3_a_norm_act)
        a = Dense(action_size, activation='linear')(drop4_a)
        a_mean = Lambda(lambda x: K.mean(x, axis=1, keepdims=True))(a)
        advantage = Subtract()([a, a_mean])
        # deuling value
        n3_v = Dense(neurons/2)(drop3)
        n3_v_norm = BatchNormalization()(n3_v)
        n3_v_norm_act = Activation('elu')(n3_v_norm)
        drop4_v = Dropout(0.3)(n3_v_norm_act)
        value = Dense(1, activation='linear')(drop4_v)
        # dueling combine
        q_out = Add()([value, advantage])
        

        return Model(inputs=[state_input, self_state_input], outputs=q_out)
