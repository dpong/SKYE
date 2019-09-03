from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Add, Subtract, Lambda, BatchNormalization, Conv1D, Flatten, MaxPooling1D, GlobalAveragePooling1D
from tensorflow.keras.models import Model
#from tensorflow.keras.activations import softmax
#from agent.noisynet import NoisyDense
import tensorflow.keras.backend as K

#Tensorflow 2.0 Beta

class Build_model():
    tf.keras.backend.set_floatx('float64')

    def build_model(self, state_size, neurons, action_size):
        
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
        # deuling advantage
        n2_a = Dense(neurons, activation='elu')(n1_norm)
        n2_a_norm = BatchNormalization()(n2_a)
        a = Dense(action_size, activation='linear')(n2_a_norm)
        a_mean = Lambda(lambda x: K.mean(x, axis=1, keepdims=True))(a)
        advantage = Subtract()([a, a_mean])
        # deuling value
        n2_v = Dense(neurons, activation='elu')(n1_norm)
        n2_v_norm = BatchNormalization()(n2_v)
        value = Dense(1, activation='linear')(n2_v_norm)
        # combine
        q = Add()([value, advantage])

        return Model(inputs=state_input, outputs=q)
        # Model是callable function

if __name__=='__main__':
    m = Build_model()
    model = m.build_model((20,7),60,4)
    model.summary()