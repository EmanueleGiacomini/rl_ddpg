"""
actor.py
"""

import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, Activation, Input
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras import optimizers
import numpy as np

class Actor(tf.keras.Model):
    def __init__(self, state_size, action_size, max_action, name='Actor'):
        super().__init__(name=name)
        self.state_size = state_size
        self.action_size = action_size

        self.max_action = max_action

        self.l1 = Dense(300, name='L1')
        self.l2 = Dense(400, name='L2')
        self.l3 = Dense(action_size, name='L3')

        with tf.device("/cpu:0"):
            self(tf.constant(np.zeros(shape=(1,)+state_size, dtype=np.float64)))

    def call(self, inputs):
        features = tf.nn.relu(self.l1(inputs))
        features = tf.nn.relu(self.l2(features))
        features = self.l3(features)
        action = self.max_action * tf.nn.tanh(features)
        return action


if __name__ == '__main__':
    ...
