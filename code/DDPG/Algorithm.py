import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")

class ReplayMemory: # 经验回放算法
    def __init__(self, capacity, transition):
        self.capacity = capacity # 经验池的容量
        self.memory = deque(maxlen=capacity) # 用deque实现经验池，deque是一个双向队列，可以从两端append和pop
        self.Transition = transition # 用于保存transition的数据结构

    def push(self, *args):
        self.memory.append(self.Transition(*args)) # 将transition存入经验池

    def sample(self, batch_size):
        return random.sample(self.memory, min(batch_size, len(self.memory))) # 从经验池中随机采样

    def __len__(self):
        return len(self.memory) # 返回经验池的长度

class Noise: # Ornstein-Uhlenbeck噪声
    def __init__(self, mu, theta, sigma, action_dim):
        self.mu = mu # 均值
        self.theta = theta # 回归速度
        self.sigma = sigma # 标准差
        self.action_dim = action_dim # 动作维度
        self.state = np.full(action_dim, mu) # 初始化状态

    def reset(self):
        self.state = np.full(self.action_dim, self.mu) # 重置状态

    def make_noise(self):
        delta = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.action_dim) # 计算噪声
        # self.state += delta
        self.state = np.clip(self.state + delta, 0, 1) # 将噪声限制在[0, 1]之间
        return self.state

class ValueNetwork(nn.Module): # 值网络，即Critic网络
    def __init__(self, hidden_size_1, hidden_size_2, input_size, action_size):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(input_size, hidden_size_1) # 输入层，输入维度为state的维度，输出维度为hidden_size_1
        self.linear2 = nn.Linear(hidden_size_1 + action_size, hidden_size_2) # 隐藏层，输入维度为hidden_size_1 + action_size，输出维度为hidden_size_2
        self.linear3 = nn.Linear(hidden_size_2, 1) # 输出层，输出维度为1

    def forward(self, state, action): # 前向传播算法
        x = F.relu(self.linear1(state))
        x = torch.cat((x, action), dim=1) # 将state和action拼接在一起
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

class PolicyNetwork(nn.Module): # 策略网络，即Actor网络
    def __init__(self, hidden_size_1, hidden_size_2, input_size, action_size):
        super(PolicyNetwork, self).__init__()

        self.linear1 = nn.Linear(input_size, hidden_size_1) # 输入层，输入维度为state的维度，输出维度为hidden_size_1
        self.linear2 = nn.Linear(hidden_size_1, hidden_size_2) # 隐藏层，输入维度为hidden_size_1，输出维度为hidden_size_2
        self.linear3 = nn.Linear(hidden_size_2, action_size) # 输出层，输出维度为action的维度

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))
        return x

