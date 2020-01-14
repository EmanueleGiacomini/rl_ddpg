"""
mountain_car_continuous_ddpg.py
"""

import gym
from ddpg import ddpg




if __name__ == '__main__':
    env = gym.make('MountainCarContinuous-v0')
    ddpg(env)

