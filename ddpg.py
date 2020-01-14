"""
ddpg.py
"""
from agent import Agent
import numpy as np
import pickle


def ddpg(env, n_episodes=2000, max_t=700):
    agent = Agent(env.observation_space.shape[0],
                  env.action_space.shape[0],
                  0)
    scores = []
    max_score = -np.inf
    for episode in range(n_episodes):
        state = env.reset()
        score = 0

        if episode % 10 == 0:
            agent.store_weights(episode)
            vectorized_scores = np.array(scores)
            np.save(f'./scores/score-{episode}', vectorized_scores)
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
