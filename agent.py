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

BUFFER_SIZE = int(10000)  # Replay buffer size
BATCH_SIZE = 128  # minibatch size
MIN_MEM_SIZE = 5000  # Minimum memory size before training
GAMMA = 0.99  # discount factor
TAU = 0.005  # soft update merge factor
LR_ACTOR = 0.002  # Actor's Learning rate
LR_CRITIC = 0.003  # Critic's Learning rate
UPDATE_STEPS = 1
# WEIGHT_DECAY = 0.0001  # L2 weight decay
CKPTS_PATH = './tf_ckpts'
ACTOR_CKPTS = 'actor'
CRITIC_CKPTS = 'critic'


class Agent(object):
    def __init__(self, state_space, action_space, max_action, device):
        self.state_size = state_space.shape[0]
        self.action_size = action_space.shape[0]
        self.max_action = max_action
        self.device = device
        self.actor_local = Actor(state_space.shape, action_space.high.size, max_action)
        self.actor_target = Actor(state_space.shape, action_space.high.size, max_action)
        self.actor_optimizer = optimizers.Adam(LR_ACTOR)
        # let target be equal to local
        self.actor_target.set_weights(self.actor_local.get_weights())

        self.critic_local = Critic(state_space.shape, action_space.high.size)
        self.critic_target = Critic(state_space.shape, action_space.high.size)
        self.critic_optimizer = optimizers.Adam(LR_CRITIC)
        # let target be equal to local
        self.critic_target.set_weights(self.critic_local.get_weights())

        self.noise = OUNoise(self.action_size)
        self.memory = ReplayBuffer(BUFFER_SIZE)

        self.current_steps = 0

    def step(self, state, action, reward, done, next_state, train=True) -> None:
        self.memory.store(state, action, reward, done, next_state)
        if train and self.memory.count > BATCH_SIZE and self.memory.count > MIN_MEM_SIZE:
            if self.current_steps % UPDATE_STEPS == 0:
                experiences = self.memory.sample(BATCH_SIZE)
                self.learn(experiences, GAMMA)
            self.current_steps += 1

    @tf.function
    def critic_train(self, states, actions, rewards, dones, next_states):
        with tf.device(self.device):
            # Compute yi
            u_t = self.actor_target(next_states)
            q_t = self.critic_target([next_states, u_t])
            yi = tf.cast(rewards, dtype=tf.float64) + \
                 tf.cast(GAMMA, dtype=tf.float64) * \
                 tf.cast((1 - tf.cast(dones, dtype=tf.int64)), dtype=tf.float64) * \
                 tf.cast(q_t, dtype=tf.float64)

            # Compute MSE
            with tf.GradientTape() as tape:
                q_l = tf.cast(self.critic_local([states, actions]), dtype=tf.float64)
                loss = (q_l - yi) * (q_l - yi)
                loss = tf.reduce_mean(loss)
                # Update critic by minimizing loss
                dloss_dql = tape.gradient(loss, self.critic_local.trainable_weights)
            self.critic_optimizer.apply_gradients(
                zip(dloss_dql, self.critic_local.trainable_weights))
        return

    @tf.function
    def actor_train(self, states):
        with tf.device(self.device):
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(self.actor_local.trainable_variables)
                u_l = self.actor_local(states)
                q_l = -tf.reduce_mean(self.critic_local([states, u_l]))
            j = tape.gradient(q_l, self.actor_local.trainable_variables)
            self.actor_optimizer.apply_gradients(
                zip(j, self.actor_local.trainable_variables))
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
        self.actor_train(states)
        self.update_local()
        return

    def update_local(self):
        def soft_updates(local_model: tf.keras.Model, target_model: tf.keras.Model) -> np.ndarray:
            local_weights = np.array(local_model.get_weights())
            target_weights = np.array(target_model.get_weights())

            assert len(local_weights) == len(target_weights)
            new_weights = TAU * local_weights + (1 - TAU) * target_weights
            return new_weights

        self.actor_target.set_weights(soft_updates(self.actor_local,
                                                   self.actor_target))
        self.critic_target.set_weights(soft_updates(self.critic_local,
                                                    self.critic_target))

    def store_weights(self, episode: int) -> None:
        self.actor_target.save_weights(join(CKPTS_PATH, ACTOR_CKPTS, f'cp-{episode}'))
        self.critic_target.save_weights(join(CKPTS_PATH, CRITIC_CKPTS, f'cp-{episode}'))
        return

    def act(self, state, add_noise=True) -> (float, float):
        state = np.array(state).reshape(1, self.state_size)
        pure_action = self.actor_local.predict(state)[0]
        action = self.noise.get_action(pure_action)
        return action, pure_action

    def reset(self):
        self.noise.reset()


if __name__ == '__main__':
    a = Agent(3, 1, 42)
