from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Add, Subtract, Lambda, BatchNormalization, concatenate, Activation
from tensorflow.keras.layers import Conv1D, Flatten, MaxPooling1D, GlobalAveragePooling1D, Embedding, Dropout
from tensorflow.keras.models import Model
from agent.noisynet import NoisyDense
import tensorflow.keras.backend as K


class Build_model():
    tf.keras.backend.set_floatx('float64')
    def build_model(self, state_size, neurons, action_size, atoms, training):
        state_input = Input(shape=state_size, name='state_input', dtype='float64')
        drop0 = Dropout(0.2)(state_input)
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
        # 連結層
        n1 = NoisyDense(neurons, Noisy=training)(drop1)
        n1_norm = BatchNormalization()(n1)
        n1_norm_act = Activation('elu')(n1_norm)
        drop2 = Dropout(0.5)(n1_norm_act)
        n2 = NoisyDense(neurons, Noisy=training)(drop2)
        n2_norm = BatchNormalization()(n2)
        n2_norm_act = Activation('elu')(n2_norm)
        drop3 = Dropout(0.5)(n2_norm_act)
        # 開始 distribution
        duel_noisydense_a_1 = []
        duel_batch_a_1 = []
        duel_activation_a_1 = []
        duel_noisydense_a_2 = []
        duel_batch_a_2 = []
        duel_activation_a_2 = []
        duel_noisydense_v_1 = []
        duel_batch_v_1 = []
        duel_activation_v_1 = []
        advantage_list=[]
        output_list=[]
        # 建立一堆 Noisy 層 with elu activations
        for i in range(action_size):
            # 第一層
            duel_noisydense_a_1.append(
                NoisyDense(atoms, Noisy=training)(drop3)
            )
            duel_batch_a_1.append(
                BatchNormalization()(duel_noisydense_a_1[i])
            )
            duel_activation_a_1.append(
                Activation('elu')(duel_batch_a_1[i])
            )
            duel_noisydense_v_1.append(
                NoisyDense(atoms, Noisy=training)(drop3)
            )
            duel_batch_v_1.append(
                BatchNormalization()(duel_noisydense_v_1[i])
            )
            duel_activation_v_1.append(
                Activation('elu')(duel_batch_v_1[i])
            )
            # 第二層
            duel_noisydense_a_2.append(
                NoisyDense(atoms, Noisy=training)(duel_activation_a_1[i])
            )
            duel_batch_a_2.append(
                BatchNormalization()(duel_noisydense_a_2[i])
            )
            duel_activation_a_2.append(
                Activation('elu')(duel_batch_a_2[i])
            )
        # deuling 計算 value     
        value_in = tf.concat([t for t in duel_activation_v_1], 1)
        value = NoisyDense(atoms, activation='elu', Noisy=training)(value_in)

        # dueling 計算 a 的 mean 值
        a_mean = tf.reduce_mean(duel_activation_a_2, 0)
        
        # (a - a_mean) + value 
        for i in range(action_size): 
            advantage_list.append(
                Subtract()([duel_activation_a_2[i], a_mean])
                )
            output_list.append(
                Add()([value, advantage_list[i]])
                )
        return Model(inputs=state_input, outputs=output_list)
