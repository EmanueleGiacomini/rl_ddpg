"""
ddpg.py
"""
from agent import Agent
import numpy as np
import pickle


def print_episode(ep, t_r, t_d, t_mean, t_std, t_s,
                  r, d, mean, std, s):
    def iteration_str(reward, done, action_mean, action_std, steps) -> str:
        return f'reward: {"%.2f" % reward}, done: {done}, action_mean: {"%.3f" % action_mean},' \
               f' action_std: {"%.3f" % action_std}, steps: {steps}'
    print(f'{ep}: Train: {iteration_str(t_r, t_d, t_mean, t_std, t_s)}\t'
          f'Test: {iteration_str(r, d, mean, std, s)}')


class DDPG(object):
    def __init__(self, env):
        self.env = env
        self.agent = Agent(self.env.observation_space.shape[0],
                           self.env.action_space.shape[0], 2)
        ...
    def run_epoch(self, max_steps, render=False, training=True):
        state = self.env.reset()
        self.agent.reset()
        actions = []
        total_reward = 0
        done = False
        for steps in range(max_steps):
            # Noise_action | Pure_action
            n_action, p_action = self.agent.act(state)
            if training:
                action = n_action
            else:
                action = p_action
            # Let the env advance
            next_state, reward, done, info = self.env.step(action)
            done = done==True
            total_reward += reward
            actions.append(action)
            # Only update the agent if we're in training phase
            self.agent.step(state, action, reward, done, next_state, training)
            if render:
                self.env.render()

            if done:
                break
        action_mean = np.mean(actions)
        action_std = np.std(actions)
        return total_reward, done, action_mean, action_std, steps

    def run(self, max_episodes: int, max_iterations: int, render: bool):
        for ep in range(max_episodes):
            train_r, train_d, train_mean, train_std, train_s = self.run_epoch(max_iterations,
                                                                              render=render)
            test_r, test_d, test_mean, test_std, test_s = self.run_epoch(max_iterations,
                                                                         render=render,
                                                                         training=False)
            print_episode(ep, train_r, train_d, train_mean, train_std, train_s,
                          test_r, test_d, test_mean, test_std, test_s)
        self.env.close()





def ddpg(env, n_episodes=2000, max_t=700):
    agent = Agent(env.observation_space.shape[0],
                  env.action_space.shape[0],
                  0)
    scores = []
    max_score = -np.inf
    for episode in range(n_episodes):
        state = env.reset()
        score = 0

        """
        if episode % 10 == 0:
            agent.store_weights(episode)
            vectorized_scores = np.array(scores)
            np.save(f'./scores/score-{episode}', vectorized_scores)
        """
        for t in range(max_t):
            env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            done = done is True
            agent.step(state, action, reward, done, next_state)
            state = next_state
            # Save scoring
            score += reward
            if done:
                break
        if max_score < score:
            max_score = score
        print(f'ep:{episode}\tscore: {score}')
        scores.append(score)
    env.close()

