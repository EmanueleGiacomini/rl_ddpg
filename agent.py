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

BUFFER_SIZE = int(1e4)  # Replay buffer size
BATCH_SIZE = 32  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 1e-3  # soft update merge factor
LR_ACTOR = 1e-4  # Actor's Learning rate
LR_CRITIC = 3e-4  # Critic's Learning rate
WEIGHT_DECAY = 0.0001  # L2 weight decay


class Agent(object):
    def __init__(self, state_size, action_size, random_seed):
        self.state_size = state_size
        self.action_size = action_size
        self.actor_local = Actor(state_size, action_size, LR_ACTOR)
        self.actor_target = Actor(state_size, action_size, LR_ACTOR)
        self.actor_optimizer = optimizers.Adam(LR_ACTOR)

        self.critic_local = Critic(state_size, action_size, LR_CRITIC)
        self.critic_target = Critic(state_size, action_size, LR_CRITIC)
        self.critic_optimizer = optimizers.Adam(LR_CRITIC)

        self.noise = OUNoise(action_size, random_seed)
        self.memory = ReplayBuffer(BUFFER_SIZE)

    def step(self, state, action, reward, done, next_state) -> None:
        self.memory.store(state, action, reward, done, next_state)
        if self.memory.count > BATCH_SIZE:
            experiences = self.memory.sample(BATCH_SIZE)
            self.learn(experiences, GAMMA)
            self.update_local()

    def learn(self, experiences, gamma) -> None:
        states, actions, rewards, dones, next_states = experiences

        states = np.array(states).reshape(BATCH_SIZE, self.state_size)
        states = tf.convert_to_tensor(states)
        actions = np.array(actions).reshape(BATCH_SIZE, self.action_size)
        actions = tf.convert_to_tensor(actions)
        rewards = np.array(rewards).reshape(BATCH_SIZE, 1)
        next_states = np.array(next_states).reshape(BATCH_SIZE, self.state_size)
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(self.critic_local.network.trainable_weights)
            tape.watch(actions)
            # -----Update Critic------------------#
            target_actor_pred = self.actor_target.network(states)
            target_critic_pred = self.critic_target.network([next_states,
                                                             target_actor_pred])

            yi = rewards + GAMMA * target_critic_pred
            # Compute the Loss MSE(yi, critic_local(states, actions)
            local_critic_pred = self.critic_local.network([states, actions])
            critic_loss = tf.keras.losses.MSE(yi, local_critic_pred)
            # Minimize the loss on the critic local
            dq_dt, dq_da = tape.gradient(critic_loss, [self.critic_local.network.trainable_weights,
                                                       actions])
            self.critic_optimizer.apply_gradients(
                zip(dq_dt, self.critic_local.network.trainable_weights))
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(self.actor_local.network.trainable_weights)
            # -----Update Actor------------------#
            local_actor_pred = self.actor_local.network(states)
            q_pred = -self.critic_local.network([states, local_actor_pred])
            j_grad = tape.gradient(q_pred, self.actor_local.network.trainable_weights)
            self.actor_optimizer.apply_gradients(zip(j_grad, self.actor_local.network.trainable_weights))

    def update_local(self):
        for target_param, local_param in zip(self.actor_target.network.trainable_weights,
                                             self.actor_local.network.trainable_weights):
            target_param = (TAU * local_param + (1.0 - TAU) * target_param)

        for target_param, local_param in zip(self.critic_target.network.trainable_weights,
                                             self.critic_local.network.trainable_weights):
            target_param = (TAU * local_param + (1.0 - TAU) * target_param)

        """s
        with tf.GradientTape(watch_accessed_variables=False) as g:
            # Update Critic #
            target_actor_pred = self.actor_target.network.predict(states)
            #print(f'actor_pred : {target_actor_pred}')
            #print(f'states shape: {states.shape}\t target_actor_pred_shape: {target_actor_pred.shape}')

            target_critic_pred = self.critic_target.network.predict([states, target_actor_pred])
            #dq_da = g.gradient(target_critic_pred, self.    )

            yi = rewards + GAMMA * target_critic_pred
            self.critic_local.network.fit([states, actions], [yi], verbose=0)

            # Update Actor #
            # Compute actor loss
            local_actor_pred = self.actor_local.network.predict(states)
            actor_loss = tf.keras.losses.MSE(yi, self.critic_local.network.predict([states, local_actor_pred]))
            #
            da_dtheta = g.gradient(actor_loss, self.actor_local.network.trainable_weights)
            """

    def act(self, state, add_noise=True):
        state = np.array(state).reshape(1, self.state_size)
        action = self.actor_local.network.predict(state)[0]
        action += self.noise.sample()
        # print(f'state: {state}\taction: {action}')
        return action


if __name__ == '__main__':
    a = Agent(3, 1, 42)
