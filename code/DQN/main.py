import gym
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
warnings.filterwarnings("ignore")
from DQN import DQN
from Agent import Agent


def train(agent):
    print('----------Start training----------')
    all_scores = [] # 记录每个episode的reward
    successful_sequences = 0 # 记录连续成功的次数
    for ep in range(1, agent.max_episode + 1):
        state = agent.env.reset() # 初始化状态
        state = torch.tensor(state).to(agent.device).float().unsqueeze(0) # 转换为tensor
        done = False
        episode_reward = 0 # 记录每个episode的reward

        while not done:
            action = agent.act(state, agent.eps) # 选择动作
            action = torch.tensor(action).to(agent.device) # 转换为tensor

            next_state, reward, done, info = agent.env.step(action.item()) # 执行动作
            episode_reward += reward # 计算reward

            modified_reward = reward + 300 * (agent.gamma * abs(next_state[1]) - abs(state[0][1])) # 修改reward

            next_state = torch.tensor(next_state).to(agent.device).float().unsqueeze(0) # 转换为tensor
            modified_reward = torch.tensor(modified_reward).to(agent.device).float().unsqueeze(0) # 转换为tensor
            done = torch.tensor(done).to(agent.device).unsqueeze(0)

            agent.memory.push(state, action, next_state, modified_reward, done) # 存储经验
            state = next_state # 更新状态

            agent.optimize() # 优化模型

        if ep % agent.target_update_rate == 0: # 更新target网络
            agent.model.update_target()

        agent.eps = max(agent.eps_end, agent.eps * agent.eps_decay) # 更新epsilon
        all_scores.append(episode_reward)

        if ep % 50 == 0: # 每50个episode输出一次平均reward
            print('episode', ep, ':', np.mean(all_scores[-50:]), 'average score')

        if np.mean(all_scores[-50:]) >= agent.goal: # 如果连续5个episode的平均reward大于-110，就认为训练成功
            successful_sequences += 1
            if successful_sequences == 5:
                print('success at episode', ep)
                # 保存模型
                # agent.save()
                return all_scores
        else: # 否则，重置连续成功的次数
            successful_sequences = 0

    return all_scores

def plot(scores):
    plt.plot(scores)
    plt.xlabel('Episode')
    plt.ylabel('Score')
    # plt.show()
    plt.savefig('./result/DQN_scores.png')

def test(agent, episodes=50, render=False):
    print('----------Start testing----------')
    state = agent.env.reset() # 初始化状态
    state = torch.tensor(state).to(agent.device).float().unsqueeze(0)
    ep_count = 0
    current_episode_reward = 0
    scores = []
    while ep_count < episodes: # 默认测试50个episode（如果不开渲染可以测试100个或更多）
        if render:
            agent.env.render()
        action = agent.act(state, 0) # 选择动作
        state, reward, done, _ = agent.env.step(action.item()) # 执行动作
        state = torch.tensor(state).to(agent.device).float().unsqueeze(0) # 转换为tensor
        current_episode_reward += reward # 计算reward

        if done: # 如果游戏结束
            ep_count += 1 # episode数加1
            scores.append(current_episode_reward) # 记录每个episode的reward
            current_episode_reward = 0 # 重置reward
            state = agent.env.reset() # 重置状态
            state = torch.tensor(state).to(agent.device).float().unsqueeze(0) # 转换为tensor

    print('average score:', sum(scores) / len(scores))
    print('max reward:', max(scores))
    print('min reward:', min(scores))

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 选择设备
    env = gym.make('MountainCar-v0')  # 创建环境
    n_actions = env.action_space.n  # 动作空间
    n_states = env.observation_space.shape[0]  # 状态空间
    Transition = namedtuple('Transition',
                            ('state', 'action', 'next_state', 'reward', 'done'))  # 定义一个命名元组，用于存储经验（状态，动作，下一个状态，奖励，是否结束）
    # 神经网络
    layers = (
        nn.Linear(n_states, 256),  # 输入层
        nn.ReLU(),  # 激活函数
        nn.Linear(256, 256),  # 隐藏层
        nn.ReLU(),  # 激活函数
        nn.Linear(256, n_actions),  # 输出层
    )
    Model = DQN(layers, lr=0.0005, optim_method=optim.Adam)  # 创建模型
    MountainCarAgent = Agent(device, Transition, env, Model, n_actions, goal=-110, min_score=-200, \
                             eps_start=1, eps_end=0.001, eps_decay=0.9, gamma=0.99, \
                             batch_size=64, memory_size=100000, max_episode=2000)

    scores = train(MountainCarAgent)
    plot(scores)
    test(MountainCarAgent, episodes=10, render=True)