from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np

# Noisy Network with C51
class NoisyDense(tf.keras.layers.Layer):
    def __init__(self, units, inputs_dim, training=True, bias=True):
        super(NoisyDense, self).__init__()
        self.training = training
        self.inputs_dim = inputs_dim
        self.units = units
        
        # mu亂數，sigma常數0.1 (我要當rainbow)
        mu_init = tf.random_uniform_initializer(minval=-1, maxval=1)
        sigma_init = tf.constant_initializer(value=0.1)
        
        # 看要不要bias , 目前都要
        if bias:  
            mu_bias_init = mu_init
            sigma_bias_init = sigma_init
        else:
            mu_bias_init = tf.zeros_initializer()
            sigma_bias_init = tf.zeros_initializer()
        
        # mu + sigma * epsilon for weight
        self.mu_w = tf.Variable(initial_value=mu_init(shape=(self.inputs_dim, self.units),
        dtype='float32'),trainable=True)
        self.sigma_w = tf.Variable(initial_value=sigma_init(shape=(self.inputs_dim, self.units),
        dtype='float32'),trainable=True)
        # mu + sigma * epsilon for bias
        self.mu_bias = tf.Variable(initial_value=mu_bias_init(shape=(self.units,),
        dtype='float32'),trainable=True)
        self.sigma_bias = tf.Variable(initial_value=sigma_bias_init(shape=(self.units,),
        dtype='float32'),trainable=True)
        
    def call(self, inputs):
        # Factor 式的 noisy
        if self.training:
            p = tf.random.normal([self.inputs_dim, self.units])
            q = tf.random.normal([51,])
            f_p = tf.multiply(tf.sign(p), tf.pow(tf.abs(p), 0.5))
            f_q = tf.multiply(tf.sign(q), tf.pow(tf.abs(q), 0.5))
            epsilon_w = f_p*f_q
            epsilon_b = tf.squeeze(f_q)
            weights = tf.add(self.mu_w, tf.multiply(self.sigma_w, epsilon_w))
            bias = tf.add(self.mu_bias, tf.multiply(self.sigma_bias, epsilon_b))
            return tf.matmul(inputs, weights) + bias
            
        else:
            weights = self.mu_w
            bias = self.mu_bias
            return tf.matmul(inputs, weights) + bias


        

    