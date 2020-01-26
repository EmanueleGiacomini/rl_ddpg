from actor import Actor
import gym
import numpy as np

from time import sleep

if __name__ == '__main__':
    env = gym.make('MountainCarContinuous-v0')
    actor = Actor(env.observation_space.shape, env.action_space.high.size, 1.)
    actor.load_weights('tf_ckpts/actor/cp-110')

    state = env.reset()
    done = False
    while not done:
        env.render()
        action = actor(np.expand_dims(state, axis=0))[0]
        next_state, reward, done, info = env.step(action)
        state = next_state
        sleep(0.05)
    sleep(1)
    env.close()
    exit(0)


