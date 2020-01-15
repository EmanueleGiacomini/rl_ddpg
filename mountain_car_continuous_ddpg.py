"""
mountain_car_continuous_ddpg.py
"""

import gym
from ddpg import ddpg


NUM_EPISODES = 3000
MAX_IT = 1000

if __name__ == '__main__':
    env = gym.make('MountainCarContinuous-v0')
    ddpg(env, n_episodes=NUM_EPISODES, max_t=MAX_IT)

