"""
mountain_car_continuous_ddpg.py
"""

import gym
from ddpg import DDPG
import numpy as np


NUM_EPISODES = 3000
MAX_IT = 950
RENDER_FLAG = False

if __name__ == '__main__':
    env = gym.make('MountainCarContinuous-v0')
    #env = gym.make('Pendulum-v0')
    ddpg_agent = DDPG(env)
    ddpg_agent.run(NUM_EPISODES, MAX_IT, RENDER_FLAG)
    #ddpg(env, n_episodes=NUM_EPISODES, max_t=MAX_IT)

