from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Add, Subtract, Lambda, BatchNormalization, concatenate, Activation
from tensorflow.keras.layers import Conv1D, Flatten, MaxPooling1D, GlobalAveragePooling1D, Embedding, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from agent.noisynet import NoisyDense
import tensorflow.keras.backend as K


class Build_model():
    tf.keras.backend.set_floatx('float64')
    def build_model(self, state_size, neurons, action_size, training):
        state_input = Input(shape=state_size, name='state_input', dtype='float64')
        drop0 = Dropout(0.2)(state_input)
        # multi-head-cnn
        con1 = Conv1D(state_size[-1]*state_size[-1], 3)(drop0)
        con1_norm_act = Activation('relu')(con1)
        drop1 = Dropout(0.5)(con1_norm_act)
        pool_max1 = MaxPooling1D(pool_size=5)(drop1)
        flat1 = Flatten()(pool_max1)

        con2 = Conv1D(state_size[-1]*state_size[-1], 5)(drop0)
        con2_norm_act = Activation('relu')(con2)
        drop2 = Dropout(0.5)(con2_norm_act)
        pool_max2 = MaxPooling1D(pool_size=5)(drop2)
        flat2 = Flatten()(pool_max2)

        con3 = Conv1D(state_size[-1]*state_size[-1], 10)(drop0)
        con3_norm_act = Activation('relu')(con3)
        drop3 = Dropout(0.5)(con3_norm_act)
        pool_max3 = MaxPooling1D(pool_size=5)(drop3)
        flat3 = Flatten()(pool_max3)
        
        merged = concatenate([flat1, flat2, flat3])
        
        # 連結層
        n1 = NoisyDense(neurons*2, Noisy=training)(merged)
        n1_norm = BatchNormalization()(n1)
        n1_norm_act = Activation('relu')(n1_norm)
        drop2 = Dropout(0.5)(n1_norm_act)
        n2 = NoisyDense(neurons, Noisy=training)(drop2)
        n2_norm = BatchNormalization()(n2)
        n2_norm_act = Activation('relu')(n2_norm)
        drop3 = Dropout(0.5)(n2_norm_act)
        # deuling advantage
        n3_a = NoisyDense(int(neurons/2), Noisy=training)(drop3)
        n3_a_norm = BatchNormalization()(n3_a)
        n3_a_norm_act = Activation('relu')(n3_a_norm)
        drop4_a = Dropout(0.3)(n3_a_norm_act)
        a = Dense(action_size, activation='linear')(drop4_a)
        a_mean = Lambda(lambda x: K.mean(x, axis=1, keepdims=True))(a)
        advantage = Subtract()([a, a_mean])
        # deuling value
        n3_v = NoisyDense(int(neurons/2), Noisy=training)(drop3)
        n3_v_norm = BatchNormalization()(n3_v)
        n3_v_norm_act = Activation('relu')(n3_v_norm)
        drop4_v = Dropout(0.3)(n3_v_norm_act)
        value = Dense(1, activation='linear')(drop4_v)
        # dueling combine
        q_out = Add()([value, advantage])
        

        return Model(inputs=state_input, outputs=q_out)
