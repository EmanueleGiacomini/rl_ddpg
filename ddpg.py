"""
ddpg.py
"""
from agent import Agent
import numpy as np


def ddpg(env, n_episodes=2000, max_t=700):
    agent = Agent(env.observation_space.shape[0],
                  env.action_space.shape[0],
                  0)
    scores = []
    max_score = -np.inf
    for episode in range(n_episodes):
        state = env.reset()
        score = 0
        for t in range(max_t):
            env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, done, next_state)
            state = next_state
            # Save scoring
            score += reward
        if max_score < score:
            max_score = score
        print(f'ep:{episode} : score : {score}')
        scores.append(score)
    env.close()
