"""
critic.py
"""

import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, Activation, Input, Concatenate, Add
import numpy as np

class Critic(tf.keras.Model):
    def __init__(self, state_shape, action_dim, name='Critic'):
        super().__init__(name=name)

        self.l1 = Dense(300, name='L1')
        self.l2 = Dense(400, name='L2')
        self.l3 = Dense(1, name='L3')

        dummy_state = tf.constant(np.zeros(shape=(1,)+state_shape, dtype=np.float64))
        dummy_action = tf.constant(np.zeros(shape=[1, action_dim], dtype=np.float64))
        with tf.device("/cpu:0"):
            self([dummy_state, dummy_action])

    def call(self, inputs):
        states, actions = inputs
        features = tf.concat([states, actions], axis=1)
        features = tf.nn.relu(self.l1(features))
        features = tf.nn.relu(self.l2(features))
        features = self.l3(features)
        return features

if __name__ == '__main__':
    ...
