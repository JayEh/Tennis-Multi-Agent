# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 17:50:29 2019

@author: jarre
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

from maddpg_agent import Agent, MADDPG
from collections import deque
from unityagents import UnityEnvironment

def new_unity_environment(train_mode=True):
    env = UnityEnvironment(file_name=".//Tennis_Windows_x86_64//Tennis.exe")
    
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    
    env_info = env.reset(train_mode=train_mode)[brain_name] # reset the environment
    state = env_info.vector_observations[0]            # get the current state
    
    # look up the size of the action and state spaces
    state_size = env_info.vector_observations[0].shape[0]
    action_size = brain.vector_action_space_size
    
    return (brain_name, env, env_info, state, state_size, action_size)


def maddpg_train(maddpg, env, brain_name, state_size, train_mode=True, n_episodes=10000, max_t=1000):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    
    for i_episode in range(1, n_episodes+1):
        # reset the environment for the start of a new episode
        env_info = env.reset(train_mode=train_mode)[brain_name] 
        
        # get the current state
        state = env_info.vector_observations
        
        # the score each episode starts at zero
        score = 0
        
        for t in range(max_t):
            actions = maddpg.act(state, add_noise=True)
            env_info = env.step(actions)[brain_name]
            next_state = env_info.vector_observations
            rewards = env_info.rewards  
            dones = env_info.local_done 
            score += max(rewards)       # take only the highest score
            maddpg.step(state, actions, rewards, next_state, dones)
            state = next_state
            if any(dones):  # if either agent is done, you're done!
                break 
            
        scores_window.append(score)       # save most recent score
        scores.append(score)              
        mean_score = np.mean(scores_window)
                
        print('\rEpisode {}\tAverage Score: {:.2f}\tActual Score:{:.2f}'.format(i_episode, mean_score, score), end="")
        if i_episode % 10 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}\tActual Score:{:.2f}'.format(i_episode, mean_score, score))
        if np.mean(scores_window)>=0.01:  # solved is 0.5
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}\tActual Score:{:.2f}'.format(i_episode, mean_score, score))
            for i, agent in enumerate(maddpg.agents):
                torch.save(agent.actor_local.state_dict(), 'Z:/{:.2f}_actor_{}_checkpoint.pth'.format(mean_score, i))
            break  # or not and just keep on keepin on
            
    return scores


brain_name, env, env_info, state, state_size, action_size = new_unity_environment(train_mode=True)
print(brain_name)
print(env)
print(env_info)
print(state)
print(state_size)
print(action_size)

maddpg = MADDPG(state_size, action_size, 1337)

scores = maddpg_train(maddpg, env, brain_name, state_size, train_mode=True)

env.close()

# plot the scores after training to a 100 episode average score of 30
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()