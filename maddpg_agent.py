# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 14:44:02 2019

@author: jarre
"""

import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

from collections import namedtuple, deque
from model import ActorNetwork, CriticNetwork

BUFFER_SIZE = 200000    # replay buffer size
BATCH_SIZE = 256        # minibatch size

GAMMA = 0.99            # discount factor
TAU = 1e-2              # for soft update of target parameters
LR = 0.001              # learning rate for the actor and critic networks

UPDATE_EVERY = 4        # how often to update the network - this was suggested on the benchmark page    -- 20
LEARN_TIMES =  16       # how many batches to sample and learn from                                     -- 10

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MADDPG():
    def __init__(self, state_size, action_size, seed=None):
        if seed is not None:
            self.seed = seed
        
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, seed)
        self.agents = [Agent(state_size, action_size, self.memory), Agent(state_size, action_size, self.memory)]
        
    def load_weights(agent_0_weights_location, agent_1_weights_location):
        # load the weights here to watch a trained agent!
        pass
        
    def get_local_actors(self):
        return [agent.actor_local for agent in self.agents]
    
    def get_target_actors(self):
        return [agent.actor_target for agent in self.agents]
    
    def act(self, state, add_noise=False):
        return [agent.act(state[agent_num], add_noise) for agent_num, agent in enumerate(self.agents)]
    
    def target_act(self, state):
        return [agent.actor_target(state[agent_num]) for agent_num, agent in enumerate(self.agents)]
    
    def learn(self, experiences, gamma):
        for agent in self.agents:
           agent.learn(experiences)

    def step(self, state, actions, rewards, next_state, dones):
        for agent_num, agent in enumerate(self.agents):
            agent.step(state[agent_num], actions[agent_num], rewards[agent_num], next_state[agent_num], dones[agent_num])


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, memory, seed=None):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        
        self.state_size = state_size
        self.action_size = action_size
        
        if seed is not None:
            self.seed = seed

        # create the local and target actor networks
        self.actor_local = ActorNetwork(state_size, action_size, seed).to(device)
        self.actor_target = ActorNetwork(state_size, action_size, seed).to(device)
        
        # create the local and target critic networks
        self.critic_local = CriticNetwork(state_size, action_size, seed).to(device)
        self.critic_target = CriticNetwork(state_size, action_size, seed).to(device)
        
        # optimizers for local actor and critic 
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR, weight_decay=0.0)
        
        # MSE loss for updating the critic
        # self.critic_loss_function = nn.MSELoss()
        self.critic_loss_function = nn.SmoothL1Loss()

        # copy the local networks weights to the target network 
        self.copy_weights_from_local_to_target()
        
        # Replay memory
        self.memory = memory
        
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        
        # init the noise class to sample from
        self.noise = GaussianNoise(self.action_size)
        
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                for _ in range(LEARN_TIMES):
                    experiences = self.memory.sample()
                    self.learn(experiences, GAMMA)
                self.soft_update_all()


    def copy_weights_from_local_to_target(self):
        # ensure that the local and target networks are initialized with the same random weights
        # or copy you saved weights after loading into local
        for target_param, param in zip(self.actor_target.parameters(), self.actor_local.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.critic_target.parameters(), self.critic_local.parameters()):
            target_param.data.copy_(param.data)

    def act(self, state, add_noise=False):
        """Returns actions for given state as per current policy. 
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        
        # get predicted actions for current state from actor network
        self.actor_local.eval()
        with torch.no_grad():
            action_values = self.actor_local(state)
        self.actor_local.train()

        # take the predicted actions and add noise, used as exploration in a continuous environment
        action_values = action_values.cpu().data.numpy()
        
        if add_noise == True:
            action_values += self.noise.sample()
        
        return action_values
        
    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        
        # unpack the experiences tuple 
        states, actions, rewards, next_states, dones = experiences
        
        # computer the loss for the actor network per the DDPG algorithm
        actor_local_predicted_actions = self.actor_local(states)
        policy_loss = -self.critic_local(states, actor_local_predicted_actions).mean()
        
        # compute the loss for the critic network per the DDPG algorithm
        predicted_Q_vals = self.critic_local(states, actions)
        predicted_actions = self.actor_target(next_states)
        Q_next = self.critic_target(next_states, predicted_actions)
        Q_targets = rewards + (gamma * Q_next * (1 - dones))
        
        critic_loss = self.critic_loss_function(predicted_Q_vals, Q_targets)
        
        # update the networks
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()
        
    
    def soft_update_all(self):
        # and soft update the target networks
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)
        
        
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter   # use percent tau local_param.data and rest target_param.data
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


# I was reading OpenAI Spinning Up and they mentioned that the OU noise wasn't much better
# than gaussian noise in practice with DDPG, so I decided to give gaussian noise a shot instead.
# the scale of the noise decreases by a minute amount every step
class GaussianNoise:
    def __init__(self, size, scale=2.0, scale_decay=0.9999):
        self.size = size
        self.scale = scale
        self.scale_decay = scale_decay
    
    def sample(self):
        noise = np.random.normal(loc=0.0, scale=self.scale, size=self.size)
        self.scale *= self.scale_decay
        
        return noise


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
    