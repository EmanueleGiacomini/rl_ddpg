"""
actor.py
"""

import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, Activation, Input
from tensorflow.keras import optimizers
import numpy as np


class Actor(object):
    def __init__(self, state_space, action_space, lr):
        def create_actor_network() -> tf.keras.Model:
            model = tf.keras.models.Sequential()
            model.add(Dense(20, input_dim=state_space))
            model.add(Activation('relu'))
            model.add(BatchNormalization())
            model.add(Dense(10))
            model.add(Activation('relu'))
            model.add(BatchNormalization())
            model.add(Dense(action_space, activation='tanh'))
            return model

        self.state_space = state_space
        self.action_space = action_space
        self.network = create_actor_network()


if __name__ == '__main__':
    ...
