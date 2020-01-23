"""
plotter.py
"""

from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import gym
from scipy.interpolate import Rbf
from actor import Actor
from critic import Critic


def actor_decision(actor, x, y):
    z = []
    for i in range(len(x)):
        row = []
        for j in range(len(y)):
            state = np.array([x[i, j], y[i, j]]).reshape(1, 2)
            action = float(actor(state)[0])
            # print(f'state: {state}\t action: {action}')
            row.append(action)
        z.append(row)
    z = np.array(z)
    return z


def critic_decision(critic, actor, x, y):
    z = []
    for i in range(len(x)):
        row = []
        for j in range(len(y)):
            state = np.array([x[i, j], y[i, j]]).reshape(1, 2)
            action = actor.predict(state)
            q_val = float(critic([state, action]))
            row.append(q_val)
        z.append(row)
    z = np.array(z)
    return z


def plot_critic_decision(env, actor_network=None, critic_network=None):
    x = np.linspace(env.observation_space.low[0],
                    env.observation_space.high[0], 5)
    y = np.linspace(env.observation_space.low[1],
                    env.observation_space.high[1], 5)
    xi = np.linspace(env.observation_space.low[0],
                     env.observation_space.high[0], 30)
    yi = np.linspace(env.observation_space.low[1],
                     env.observation_space.high[1], 30)

    xi, yi = np.meshgrid(xi, yi)
    x, y = np.meshgrid(x, y)
    critic_output = critic_decision(critic_network, actor_network, x, y)

    rbf = Rbf(x, y, critic_output, function='linear')
    zi = rbf(xi, yi)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.imshow(zi, vmin=-1., vmax=1., origin='lower',
               extent=[xi.min(), xi.max(), yi.min(), yi.max()])
    plt.colorbar()
    ax.set_aspect('auto')
    plt.show()


def plot_actor_decision(env, actor_network=None):
    x = np.linspace(env.observation_space.low[0],
                    env.observation_space.high[0], 5)
    y = np.linspace(env.observation_space.low[1],
                    env.observation_space.high[1], 5)

    xi = np.linspace(env.observation_space.low[0],
                     env.observation_space.high[0], 30)
    yi = np.linspace(env.observation_space.low[1],
                     env.observation_space.high[1], 30)

    xi, yi = np.meshgrid(xi, yi)
    x, y = np.meshgrid(x, y)
    actor_output = actor_decision(actor_network, x, y)

    rbf = Rbf(x, y, actor_output, function='linear')
    zi = rbf(xi, yi)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.imshow(zi, vmin=-1., vmax=1., origin='lower',
               extent=[xi.min(), xi.max(), yi.min(), yi.max()])
    plt.colorbar()
    ax.set_aspect('auto')
    plt.show()


def plot_ddpg_decisions(ep: int,
                        actor_network: tf.keras.Model,
                        critic_network: tf.keras.Model,
                        env: gym.Env) -> None:
    def build_grid(resolution: int):
        x = np.linspace(env.observation_space.low[0],
                        env.observation_space.high[0], resolution)
        y = np.linspace(env.observation_space.low[1],
                        env.observation_space.high[1], resolution)
        return np.meshgrid(x, y)

    x, y = build_grid(8)
    xi, yi = build_grid(30)
    actor_vals = actor_decision(actor_network, x, y)
    critic_vals = critic_decision(critic_network, actor_network, x, y)

    rbf = Rbf(x, y, actor_vals, function='linear')
    actor_zi = rbf(xi, yi)
    rbf = Rbf(x, y, critic_vals, function='linear')
    critic_zi = rbf(xi, yi)

    fig, axs = plt.subplots(1, 2)
    fig.suptitle(f'Episode: {ep}')
    # Actor subplot
    axs[0].set_title('PI(s)')
    pi_img = axs[0].imshow(actor_zi, vmin=-.1, vmax=1.,
                           origin='lower',
                           extent=[xi.min(), xi.max(), yi.min(), yi.max()])

    axs[0].set_aspect('auto')

    axs[1].set_title('Q(s, PI(s))')
    q_img = axs[1].imshow(critic_zi, vmin=-100, vmax=100,
                          origin='lower',
                          extent=[xi.min(), xi.max(), yi.min(), yi.max()])
    axs[1].set_aspect('auto')
    plt.savefig(f'decision-plots/{ep}.png')
    return


if __name__ == '__main__':
    plt.ion()
    fig = plt.figure()
    plt.axis([0, 1000, 0, 1])

    env = gym.make('MountainCarContinuous-v0')
    actor = Actor(env.observation_space.shape[0],
                  env.action_space.shape[0], 0.0)
    critic = Critic(env.observation_space.shape[0],
                    env.action_space.shape[0], 0.0)
    # plot_actor_decision(env, actor.network)
    # plot_critic_decision(env, actor.network, critic.network)
    plot_ddpg_decisions(12, actor.network,
                        critic.network,
                        env)
