"""
agent.py
"""

from actor import Actor
from critic import Critic
from ounoise import OUNoise
from replay_buffer import ReplayBuffer
import tensorflow as tf
from tensorflow.keras import optimizers
import numpy as np
from os.path import join

BUFFER_SIZE = int(1e5)  # Replay buffer size
BATCH_SIZE = 64  # minibatch size
MIN_MEM_SIZE = 2000  # Minimum memory size before training
GAMMA = 0.99  # discount factor
TAU = 0.001  # soft update merge factor
LR_ACTOR = 0.02  # Actor's Learning rate
LR_CRITIC = 0.005  # Critic's Learning rate
WEIGHT_DECAY = 0.0001  # L2 weight decay
CKPTS_PATH = './tf_ckpts'
ACTOR_CKPTS = 'actor'
CRITIC_CKPTS = 'critic'


class Agent(object):
    def __init__(self, state_size, action_size, random_seed=0):
        self.state_size = state_size
        self.action_size = action_size
        self.actor_local = Actor(state_size, action_size, LR_ACTOR)
        self.actor_target = Actor(state_size, action_size, LR_ACTOR)
        self.actor_optimizer = optimizers.SGD(LR_ACTOR)
        # let target be equal to local
        self.actor_target.network.set_weights(self.actor_local.network.get_weights())

        self.critic_local = Critic(state_size, action_size, LR_CRITIC)
        self.critic_target = Critic(state_size, action_size, LR_CRITIC)
        self.critic_optimizer = optimizers.Adam(LR_CRITIC)
        # let target be equal to local
        self.critic_target.network.set_weights(self.critic_local.network.get_weights())

        self.noise = OUNoise(action_size)
        self.memory = ReplayBuffer(BUFFER_SIZE)

    def step(self, state, action, reward, done, next_state, train=True) -> None:
        self.memory.store(state, action, reward, done, next_state)
        if train and self.memory.count > BATCH_SIZE and self.memory.count > MIN_MEM_SIZE:
            experiences = self.memory.sample(BATCH_SIZE)
            self.learn(experiences, GAMMA)
            self.update_local()

    def critic_train(self, states, actions, rewards, dones, next_states):
        # Compute yi
        u_t = self.actor_target.network(next_states)
        q_t = self.critic_target.network([next_states, u_t])
        yi = rewards + GAMMA * (1 - dones) * q_t
        # Compute MSE
        with tf.GradientTape() as tape:
            q_l = self.critic_local.network([states, actions])
            loss = tf.keras.losses.MSE(yi, q_l)
            # Update critic by minimizing loss
            dloss_dql = tape.gradient(loss, self.critic_local.network.trainable_weights)
        self.critic_optimizer.apply_gradients(
            zip(dloss_dql, self.critic_local.network.trainable_weights))
        return

    def actor_train(self, states, actions, rewards, dones, next_states):
        with tf.GradientTape() as tape:
            u_l = self.actor_local.network(states)
            q_l = self.critic_local.network([states, u_l])
            q_l *= -1
            j = tape.gradient(q_l, self.actor_local.network.trainable_weights)
        for i in range(len(j)):
            j[i] /= BATCH_SIZE
        self.actor_optimizer.apply_gradients(
            zip(j, self.actor_local.network.trainable_weights))
        return

    def learn(self, experiences, gamma) -> None:
        states, actions, rewards, dones, next_states = experiences

        states = np.array(states).reshape(BATCH_SIZE, self.state_size)
        states = tf.convert_to_tensor(states)
        actions = np.array(actions).reshape(BATCH_SIZE, self.action_size)
        actions = tf.convert_to_tensor(actions)
        rewards = np.array(rewards).reshape(BATCH_SIZE, 1)
        next_states = np.array(next_states).reshape(BATCH_SIZE, self.state_size)
        dones = np.array(dones).reshape(BATCH_SIZE, 1)

        self.critic_train(states, actions, rewards, dones, next_states)
        self.actor_train(states, actions, rewards, dones, next_states)
        return

    def update_local(self):
        def soft_updates(local_model: tf.keras.Model, target_model: tf.keras.Model) -> np.ndarray:
            local_weights = np.array(local_model.get_weights())
            target_weights = np.array(target_model.get_weights())

            assert len(local_weights) == len(target_weights)
            new_weights = TAU * local_weights + (1 - TAU) * target_weights
            return new_weights

        self.actor_target.network.set_weights(soft_updates(self.actor_local.network,
                                                           self.actor_target.network))
        self.critic_target.network.set_weights(soft_updates(self.critic_local.network,
                                                            self.critic_target.network))

    def store_weights(self, episode: int) -> None:
        self.actor_target.network.save_weights(join(CKPTS_PATH, ACTOR_CKPTS, f'cp-{episode}'))
        self.critic_target.network.save_weights(join(CKPTS_PATH, CRITIC_CKPTS, f'cp-{episode}'))
        return

    def act(self, state, add_noise=True) -> (float, float):
        state = np.array(state).reshape(1, self.state_size)
        pure_action = self.actor_local.network.predict(state)[0]
        action = self.noise.get_action(pure_action)
        return action, pure_action

    def reset(self):
        self.noise.reset()


if __name__ == '__main__':
    a = Agent(3, 1, 42)
