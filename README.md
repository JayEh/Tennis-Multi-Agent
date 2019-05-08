[//]: # (Image References)
[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"


# Project 3: Collaboration and Competition

### Introduction

For this project, I trained an agent to solve the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.

![Trained Agent][image1]

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically, we take the score of the agent that scored the highest.

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.

### Solving the Environment

This problem presents a continuous control task where there are 2 agents competing against each other to pass a ball back and forth. To train these agents to compete I implemented an algorithm similar to MADDPG, based on the code in a lab earlier in this course, but I kept the same learning logic as the vanilla DDPG agent for both agents that are being trained here, as opposed to building and passing additional state to the critic networks as is done in the MADDPG paper. I've taken some inspiration from the MADDPG algorithm for my solution, and strictly speaking my solution is not truly MADDPG, but very close to it.

### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
    
2.  Place the downloaded file in the root folder of project and update the training script to point to this location.

### Instructions
In order to run the scripts and train an agent you will need to install a few dependencies. We need Pytorch to build our neural nets, UnityAgents to interact with the 3D Unity environment.

Numpy - https://www.scipy.org/scipylib/download.html
Numpy provides a very useful N-dimensional array object. Visit the Numpy download page for installation instructions.

Pytorch - https://pytorch.org/
Pytorch is the library used to build our neural nets. The full list of install options is on Pytorch home page.

Or if you're on Windows with Python 3.6, and CUDA 9 (like me) you can easily use pip to install Pytorch
pip3 install https://download.pytorch.org/whl/cu90/torch-1.0.1-cp36-cp36m-win_amd64.whl
pip3 install torchvision

Unity ML-Agents - https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md
You'll need Unity ML-Agents, which is the API that allows us to to communicate with the standalone Unity environment. The install options are on the Unity ml-agents Github page.

Matplotlib - https://matplotlib.org/users/installing.html#installing-an-official-release
To graph the learning history of the agent we'll use Matplotlib. Follow the link for installation instructions.

### Check out Report.ipynb in the repo to train an agent, and watch a trained agent!
If the required dependencies are installed in your environment, then just clone this repo, open Report.ipynb and hit Go! 