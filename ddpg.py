"""
ddpg.py
"""
from agent import Agent
import numpy as np
import pickle
from plotter import plot_ddpg_decisions, plot_reward


def print_episode(ep, t_r, t_d, t_mean, t_std, t_s,
                  r, d, mean, std, s):
    def iteration_str(reward, done, action_mean, action_std, steps) -> str:
        return f'reward: {"%.2f" % reward}, done: {done}, action_mean: {"%.3f" % action_mean},' \
               f' action_std: {"%.3f" % action_std}, steps: {steps}'

    print(f'{ep}: Train: {iteration_str(t_r, t_d, t_mean, t_std, t_s)}\t'
          f'Test: {iteration_str(r, d, mean, std, s)}')


class DDPG(object):
    def __init__(self, env, device):
        self.env = env
        self.agent = Agent(self.env.observation_space,
                           self.env.action_space, env.action_space.high,
                           device)
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
            done = done == True
            total_reward += reward
            actions.append(action)
            # Only update the agent if we're in training phase
            self.agent.step(state, action, reward, done, next_state, training)
            if render:
                self.env.render()
            state = next_state
            if done:
                break
        action_mean = np.mean(actions)
        action_std = np.std(actions)
        return total_reward, done, action_mean, action_std, steps

    def run(self, max_episodes: int, max_iterations: int, render: bool):
        rewards = []
        for ep in range(max_episodes):
            train_r, train_d, train_mean, train_std, train_s = self.run_epoch(max_iterations,
                                                                              render=render)
            test_r, test_d, test_mean, test_std, test_s = self.run_epoch(max_iterations,
                                                                         render=render,
                                                                         training=False)
            print_episode(ep, train_r, train_d, train_mean, train_std, train_s,
                          test_r, test_d, test_mean, test_std, test_s)

            rewards.append(test_r)

            if ep % 2 == 0 and ep > 0:
                plot_ddpg_decisions(ep,
                                    self.agent.actor_local,
                                    self.agent.critic_local,
                                    self.env)
                plot_reward(ep, list(zip(range(0, ep+2), rewards)))
                self.agent.store_weights(ep)
        self.env.close()
