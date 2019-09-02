from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Add, Subtract, Lambda, BatchNormalization, GRU # Conv1D, Flatten, MaxPooling1D, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import softmax
from agent.noisynet import NoisyDense
import tensorflow.keras.backend as K

#Tensorflow 2.0 Beta

class Build_model():
    def build_model(self, state_size, neurons, action_size, training):
        
        state_input = Input(shape=state_size)
        # GRU 層
        gru1 = GRU(neurons, activation='elu',return_sequences=True)(state_input)
        gru1_norm = BatchNormalization()(gru1)
        gru2 = GRU(neurons, activation='elu',return_sequences=True)(gru1_norm)
        gru2_norm = BatchNormalization()(gru2)
        gru3 = GRU(neurons, activation='elu',return_sequences=True)(gru2_norm)
        gru3_norm = BatchNormalization()(gru3)
        gru4 = GRU(neurons, activation='elu',return_sequences=False)(gru3_norm)
        gru4_norm = BatchNormalization()(gru4)
        # 連結層
        d2 = Dense(neurons, activation='elu')(gru4_norm)
        d2_norm = BatchNormalization()(d2)
        #dueling
        d3_a = NoisyDense(neurons, neurons, activation='elu', Noisy=training, bias=True)(d2_norm)
        d3_a_norm = BatchNormalization()(d3_a)
        d3_v = NoisyDense(neurons, neurons, activation='elu', Noisy=training, bias=True)(d2_norm)
        d3_v_norm = BatchNormalization()(d3_v)
        a = NoisyDense(action_size, neurons, activation='linear', Noisy=training, bias=True)(d3_a_norm)
        value = Dense(1, activation='linear')(d3_v_norm)
        a_mean = Lambda(lambda x: K.mean(x, axis=1, keepdims=True))(a)
        advantage = Subtract()([a, a_mean])
        q = Add()([value, advantage])

        # 最後compile
        model = Model(inputs=state_input, outputs=q)
        model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001, clipnorm=0.001))
        
        return model
    
        '''
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
        '''