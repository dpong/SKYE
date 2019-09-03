from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Add, Subtract, Lambda, BatchNormalization, Conv1D, Flatten, MaxPooling1D, GlobalAveragePooling1D
from tensorflow.keras.models import Model
#from agent.noisynet import NoisyDense
import tensorflow.keras.backend as K

#Tensorflow 2.0 Beta

class Build_model():
    tf.keras.backend.set_floatx('float64')
    def build_model(self, state_size, neurons, action_size, atoms):
        state_input = Input(shape=state_size)
        norm = BatchNormalization()(state_input)  # 輸入標準化
        # 卷積層們
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
        flat = Flatten()(avg_norm)
        flat_norm = BatchNormalization()(flat)
        # 連結層
        n1 = Dense(neurons, activation='elu')(flat_norm)
        n1_norm = BatchNormalization()(n1)
        # 開始 distribution
        distribution_list_a = []
        distribution_list_v = []
        norm_list_a = []
        norm_list_v = [] 
        duel_distribution_list_a = []
        duel_norm_list_a = []
        advantage_list=[]
        output_list=[]
        # 建立一堆 Noisy 層 with elu activations
        for i in range(action_size):
            distribution_list_a.append(
                Dense(atoms, activation='elu')(n1_norm)
                )
            norm_list_a.append(
                BatchNormalization()(distribution_list_a[i])
            )
            distribution_list_v.append(
                Dense(atoms, activation='elu')(n1_norm)
                )
            norm_list_v.append(
                BatchNormalization()(distribution_list_v[i])
            )
            duel_distribution_list_a.append(
                Dense(atoms, activation='elu')(norm_list_a[i])
                )
            duel_norm_list_a.append(
                BatchNormalization()(duel_distribution_list_a[i])
            )
        # deuling 計算 value     
        value_in = tf.concat([t for t in norm_list_v], 1)
        value = Dense(atoms, activation='elu')(value_in)
        value_norm = BatchNormalization()(value)
        # Output shape is (None, atoms) 

        # dueling 計算 a 的 mean 值
        a_mean = tf.reduce_mean(duel_norm_list_a, 0)
        
        # (a - a_mean) + value 
        for i in range(action_size): 
            advantage_list.append(
                Subtract()([duel_norm_list_a[i], a_mean])
                )
            output_list.append(
                Add()([value_norm, advantage_list[i]])
                )
        
        return Model(inputs=state_input, outputs=output_list)
        # softmax 在外面加上

if __name__=='__main__':
    m = Build_model()
    model = m.build_model((20,7),60,4,51)
    model.summary()
