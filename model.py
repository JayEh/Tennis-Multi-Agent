# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 14:40:26 2019

@author: jarre
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import floor

def conv_output(input_w, filter_w, stride_w):
    return ((input_w - filter_w) / stride_w) +1
def pool_output(input_w, stride_w):
    return floor(input_w / stride_w) 
def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class CriticNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed=None, hidden=128):
        super(CriticNetwork, self).__init__()
        
        if seed is not None:
            self.seed = torch.manual_seed(seed)
        
        self.fc1 = nn.Linear(state_size, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.fc2 = nn.Linear(hidden+action_size, hidden)
        self.fc3 = nn.Linear(hidden, hidden)
        self.fc4 = nn.Linear(hidden, 1)
        self.reset_params()

    def reset_params(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, actions): # state, otherwise known as x
        """Build a network that maps state -> value """
        state   = F.relu(self.bn1(self.fc1(state)))
        state   = torch.cat((state, actions), dim=1) # cat on dim 1 because dim 0 is the batch dim
        state   = F.relu(self.fc2(state))
        state   = F.relu(self.fc3(state))
        value = self.fc4(state)
        return value


class ActorNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed, hidden=128):
        super(ActorNetwork, self).__init__()
        
        if seed is not None:
            self.seed = torch.manual_seed(seed)
        
        self.fc1 = nn.Linear(state_size, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, action_size)
        self.reset_params()

    def reset_params(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state): # state, otherwise known as x
        """Build a network that maps state -> action values."""
        state   = F.relu(self.bn1(self.fc1(state)))
        state   = F.relu(self.fc2(state))
        actions = self.fc3(state)
        norm = torch.norm(actions)
        return 1.0 * torch.tanh(norm) * actions/norm if norm > 0 else 1.0 * actions
                
        return actions