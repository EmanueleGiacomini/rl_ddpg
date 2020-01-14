"""
critic.py
"""

import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, Activation, Input, Add
from tensorflow.keras import optimizers


class Critic(object):
    def __init__(self, state_space, action_space, lr):
        def create_critic_network():
            state_input = Input(shape=(state_space,))
            action_input = Input(shape=(action_space,))
            x = Dense(400)(state_input)
            x = Activation('relu')(x)
            x = Dense(300)(x)
            y = Dense(300)(action_input)
            x = Add()([x, y])
            x = Activation('relu')(x)
            x = Dense(1)(x)
            model = tf.keras.Model(inputs=[state_input, action_input], outputs=[x])
            return model
        """
        def create_critic_network():
            state_input = Input(shape=[None, state_space])
            action_input = Input(shape=[None, action_space])
            x = Dense(400)(state_input)
            x = Activation('relu')(x)
            x = Dense(300)(x)
            y = Dense(300)(action_input)
            x = Add()([x, y])
            x = Activation('relu')(x)
            x = Dense(1)(x)
            opt = optimizers.Adam(learning_rate=lr)
            model = tf.keras.Model(inputs=[state_input, action_input], outputs=[x])
            model.compile(opt, 'mse')
            tf.keras.utils.plot_model(model, 'model.png')
            return model
        """
        self.state_space = state_space
        self.action_space = action_space
        self.network = create_critic_network()


class CriticNetwork:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim

        def create_network() -> tf.keras.Model:
            state_input = Input(shape=[None, self.state_dim])
            action_input = Input(shape=[None, self.action_dim])
            x = Dense(400)(state_input)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Dense(300)(x)
            y = Dense(300)(action_input)
            x = Add()([x, y])
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Dense(1,
                      kernel_initializer=tf.keras.initializers.RandomUniform(-0.003, 0.003))(x)
            model = tf.keras.Model(inputs=[state_input, action_input], outputs=[x])
            return model

        self.network = create_network()
        ...


if __name__ == '__main__':
    critic = CriticNetwork(2, 1)
