"""
plotter.py
"""

from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import gym
from scipy.interpolate import Rbf
from actor import Actor


def actor_decision(actor, x, y):
    z = []
    for i in range(len(x)):
        row = []
        for j in range(len(y)):
            state = np.array([x[i, j], y[i, j]]).reshape(1, 2)
            action = float(actor(state)[0])
            print(f'state: {state}\t action: {action}')
            row.append(action)
        z.append(row)
    z = np.array(z)
    return z

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


if __name__ == '__main__':
    env = gym.make('MountainCarContinuous-v0')
    actor = Actor(env.observation_space.shape[0],
                  env.action_space.shape[0], 0.0)
    plot_actor_decision(env, actor.network)
