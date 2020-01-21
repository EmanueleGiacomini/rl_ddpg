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

BUFFER_SIZE = int(5000)  # Replay buffer size
BATCH_SIZE = 64  # minibatch size
MIN_MEM_SIZE = 1000  # Minimum memory size before training
GAMMA = 0.99  # discount factor
TAU = 0.001  # soft update merge factor
LR_ACTOR = 0.0001  # Actor's Learning rate
LR_CRITIC = 0.001  # Critic's Learning rate
UPDATE_STEPS = 1
WEIGHT_DECAY = 0.0001  # L2 weight decay
CKPTS_PATH = './tf_ckpts'
ACTOR_CKPTS = 'actor'
CRITIC_CKPTS = 'critic'


class Agent(object):
    def __init__(self, state_space, action_space, random_seed=0):
        self.state_size = state_space.shape[0]
        self.action_size = action_space.shape[0]
        self.actor_local = Actor(self.state_size, self.action_size, LR_ACTOR)
        self.actor_target = Actor(self.state_size, self.action_size, LR_ACTOR)
        self.actor_optimizer = optimizers.Adam(LR_ACTOR)
        # let target be equal to local
        self.actor_target.network.set_weights(self.actor_local.network.get_weights())

        self.critic_local = Critic(self.state_size, self.action_size, LR_CRITIC)
        self.critic_target = Critic(self.state_size, self.action_size, LR_CRITIC)
        self.critic_optimizer = optimizers.Adam(LR_CRITIC)
        # let target be equal to local
        self.critic_target.network.set_weights(self.critic_local.network.get_weights())

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

    def critic_train(self, states, actions, rewards, dones, next_states):
        # Compute yi
        u_t = self.actor_target.network(next_states)
        q_t = self.critic_target.network([next_states, u_t])
        yi = rewards + GAMMA * (1 - dones) * q_t

        # Compute MSE
        with tf.GradientTape() as tape:
            q_l = self.critic_local.network([states, actions])
            loss = tf.keras.losses.MSE(yi, q_l) / BATCH_SIZE
            # Update critic by minimizing loss
            dloss_dql = tape.gradient(loss, self.critic_local.network.trainable_weights)
        self.critic_optimizer.apply_gradients(
            zip(dloss_dql, self.critic_local.network.trainable_weights))
        """
        for i in range(len(dones)):
            if dones[i]:
                print('critic_train:')
                print(f'yi_mean: {np.mean(yi)}, yi_std: {np.std(yi)}'
                      f'ql_mean: {np.mean(q_l)}, ql_std: {np.std(q_l)}')
                print('-----------')
        """
        return

    def super_actor_train(self, states, actions, rewards, dones, next_states):
        j_grad = None
        for i in range(BATCH_SIZE):
            with tf.GradientTape(watch_accessed_variables=False) as tape1:
                tape1.watch(self.actor_local.network.trainable_variables)
                pi = self.actor_local.network(states[i:i+1])
            with tf.GradientTape(watch_accessed_variables=False) as tape2:
                tape2.watch(pi)
                q = -self.critic_local.network([states[i:i+1], pi])
            q_grad = tape2.gradient(q, pi)
            pi_grad = tape1.gradient(pi, self.actor_local.network.trainable_variables, q_grad)
            if i == 0:
                j_grad = pi_grad
            else:
                for j in range(len(pi_grad)):
                    j_grad[j] += pi_grad[j]
        for j in range(len(j_grad)):
            j_grad[j] /= BATCH_SIZE
        self.actor_optimizer.apply_gradients(
            zip(j_grad, self.actor_local.network.trainable_variables)
        )


    def actor_train(self, states, actions, rewards, dones, next_states):
        with tf.GradientTape(watch_accessed_variables=False) as tape1:
            tape1.watch(self.actor_local.network.trainable_variables)
            u_l = self.actor_local.network(states)
        with tf.GradientTape(watch_accessed_variables=False) as tape2:
            tape2.watch(u_l)
            q_l = self.critic_local.network([states, u_l])
        q_grads = tape2.gradient(q_l, u_l)
        j = tape1.gradient(u_l, self.actor_local.network.trainable_variables, -q_grads)
        #for i in range(len(j)):
        #    j[i] /= BATCH_SIZE
        self.actor_optimizer.apply_gradients(
            zip(j, self.actor_local.network.trainable_variables))
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
        #self.super_actor_train(states, actions, rewards, dones, next_states)
        self.update_local()
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
