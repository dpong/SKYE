from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Add, Subtract, Lambda, BatchNormalization, Conv1D, Flatten, MaxPooling1D, GlobalAveragePooling1D
from tensorflow.keras.models import Model
#from agent.noisynet import NoisyDense
import tensorflow.keras.backend as K
from tensorflow.nn import softmax

#Tensorflow 2.0 Beta

class Build_model():
    tf.keras.backend.set_floatx('float64')
    def build_model(self, state_size, neurons, action_size, atoms, training):
        state_input = Input(shape=state_size)
        norm = BatchNormalization()(state_input)  # 輸入標準化
        # 卷積層們
        con1 = Conv1D(neurons, state_size[1], padding="same", activation='relu')(norm)
        con_norm1 = BatchNormalization()(con1)
        con2 = Conv1D(neurons, state_size[1], padding="same", activation='relu')(con_norm1)
        con_norm2 = BatchNormalization()(con2)
        con3 = Conv1D(neurons, state_size[1], padding="same", activation='relu')(con_norm2)
        con_norm3 = BatchNormalization()(con3)
        con4 = Conv1D(neurons, state_size[1], padding="same", activation='relu')(con_norm3)
        con_norm4 = BatchNormalization()(con4)
        flat = Flatten()(con_norm4)
        # 開始 distribution
        #distribution_list_a = []
        #distribution_list_v = []
        #norm_list_a = []
        #norm_list_v = [] 
        #duel_distribution_list_a = []
        #duel_norm_list_a = []
        #advantage_list=[]
        output_list=[]
        for i in range(action_size):
            output_list.append(
                Dense(atoms, activation='linear')(flat)
                )

        return Model(inputs=state_input, outputs=output_list)


        '''
        # 建立一堆 Noisy 層 with elu activations
        for i in range(action_size):
            distribution_list_a.append(
                Dense(atoms, neurons, activation='relu')(n1)
                )
            #norm_list_a.append(
            #    BatchNormalization()(distribution_list_a[i])
            #)
            distribution_list_v.append(
                Dense(atoms, neurons, activation='relu')(n1)
                )
            #norm_list_v.append(
            #    BatchNormalization()(distribution_list_v[i])
            #)
            duel_distribution_list_a.append(
                Dense(atoms, atoms, activation='linear')(distribution_list_a[i])
                )
            #duel_norm_list_a.append(
            #    BatchNormalization()(duel_distribution_list_a[i])
            #)
        # deuling 計算 value     
        value_in = tf.concat([t for t in distribution_list_v], 1)
        value = Dense(atoms, activation='linear')(value_in)
        
        # Output shape is (None, atoms) 

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
        '''