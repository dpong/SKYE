from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Add, Subtract, Lambda, BatchNormalization, concatenate
from tensorflow.keras.layers import Conv1D, Flatten, MaxPooling1D, GlobalAveragePooling1D, Embedding
from tensorflow.keras.models import Model
#from agent.noisydense import NoisyDense
import tensorflow.keras.backend as K


class Build_model():
    tf.keras.backend.set_floatx('float64')
    def build_model(self, state_size, self_feat_shape, neurons, action_size, atoms):
        state_input = Input(shape=state_size, name='state_input', dtype='float64')
        norm = BatchNormalization()(state_input)  # 輸入標準化
        # 額外輸入
        self_state_input = Input(shape=(self_feat_shape[-2], self_feat_shape[-1]), name='self_state_input', dtype='float64')
        s1 = Embedding(self_feat_shape[-1], self_feat_shape[-1], input_length=self_feat_shape[-2])(self_state_input)
        flat_s1 = Flatten()(s1)
        flat_norm_s1 = BatchNormalization()(flat_s1)
        # 卷積層們，kernel_size為5天，一週的概念
        con1 = Conv1D(state_size[1], 5, padding='same', activation='relu')(norm)
        con2 = Conv1D(state_size[1], 5, padding='same', activation='relu')(con1)
        pool_max = MaxPooling1D(pool_size=5, strides=1, padding='same')(con2)
        flat = Flatten()(pool_max)
        flat_norm = BatchNormalization()(flat)
        # 外插 self_state_input
        connect = concatenate([flat_norm, flat_norm_s1])
        # 連結層
        n1 = Dense(neurons, activation='relu', name='n1')(connect)
        n1_norm = BatchNormalization()(n1)
        # 開始 distribution
        duel_distribution_list_a = []
        advantage_list=[]
        output_list=[]
        # 建立一堆 Noisy 層 with elu activations
        for i in range(action_size):
            duel_distribution_list_a.append(
                Dense(atoms, activation='linear', name='a_{}'.format(i))(n1_norm)
                )
        # deuling 計算 value     
        #value_in = tf.concat([t for t in norm_list_v], 1)
        value = Dense(atoms, activation='elu', name='value')(n1_norm)

        # dueling 計算 a 的 mean 值
        a_mean = tf.reduce_mean(duel_distribution_list_a, 0)
        
        # (a - a_mean) + value 
        for i in range(action_size): 
            advantage_list.append(
                Subtract()([duel_distribution_list_a[i], a_mean])
                )
            output_list.append(
                Add()([value, advantage_list[i]])
                )
        return Model(inputs=[state_input, self_state_input], outputs=output_list)
