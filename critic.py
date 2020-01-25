"""
critic.py
"""

import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, Activation, Input, Concatenate, Add
from tensorflow.keras import optimizers


class Critic(object):
    def __init__(self, state_space, action_space, lr):
        def create_critic_network():
            state_in = Input(shape=(state_space,), dtype='float64')
            state_net = BatchNormalization()(state_in)
            state_net = Dense(32, activation='relu')(state_net)

            action_in = Input(shape=(action_space,), dtype='float64')
            action_net = Dense(32, activation='relu')(action_in)
            net = Add()([state_net, action_net])
            out = Dense(1, activation='linear')(net)
            model = tf.keras.Model(inputs=[state_in, action_in], outputs=[out])
            return model

        self.state_space = state_space
        self.action_space = action_space
        self.network = create_critic_network()

if __name__ == '__main__':
    ...
