from DDPG import DDPG
from Agent import Agent
from Algorithm import Noise
import gym
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
import torch
import torch.optim as optim
import torch.nn.functional as F
import sys
import pickle
import warnings
warnings.filterwarnings("ignore")



def train(agent):
    print('----------Start training----------')
    all_scores = [] # 保存每个episode的reward
    successful_sequences = 0 # 连续成功的次数
    step = 0 # 记录训练的步数
    eps = 1 # epsilon-greedy策略中的epsilon

    for ep in range(1, agent.max_episode + 1):
        state = agent.env.reset() # 初始化环境
        state = torch.tensor(state).to(agent.device).float().unsqueeze(0) # 将state转换为tensor
        done = False # 记录训练是否结束
        episode_reward = 0 # 记录每个episode的reward

        while not done: # 在一个episode中不断循环，直到done
            if ep > agent.exploration_episodes: # 前几个episode使用epsilon-greedy策略
                action = agent.act(state, eps) # 选择动作
            else: # 后面的episode使用随机策略
                action = torch.tensor([np.random.uniform(agent.env.action_space.low[0], agent.env.action_space.high[0])]) # 随机选择动作
                action = action.unsqueeze(0) # 将动作转换为tensor
            action = torch.tensor(action).to(agent.device) # 将动作转换为tensor
            next_state, reward, done, info = agent.env.step(action) # 执行动作，返回下一个状态、reward、是否结束、以及info
            episode_reward += reward # 累加reward
            modified_reward = reward + 10 * abs(next_state[1]) # 修改reward，使得agent更快地学习到正确的策略

            next_state = torch.tensor(next_state).to(agent.device).float().unsqueeze(0) # 将next_state转换为tensor
            modified_reward = torch.tensor(modified_reward).to(agent.device).float().unsqueeze(0) # 将modified_reward转换为tensor
            done = torch.tensor(done).to(agent.device).unsqueeze(0) # 将done转换为tensor

            agent.memory.push(state, action, next_state, modified_reward, done) # 将transition存入memory经验池
            state = next_state # 更新state为下一个状态
            agent.optimize() # 优化网络参数

            step += 1
            if step % agent.target_update_rate == 0: # 每隔一定步数更新target网络
                agent.Model.update_target_networks()

            if done:
                if episode_reward > agent.min_score: # 如果reward大于min_score，则输出reward和episode
                    print(episode_reward, 'at episode', ep)

        eps = max(eps * 0.95, 0.1) # 将epsilon-greedy策略中的epsilon逐渐减小
        all_scores.append(episode_reward) # 保存每个episode的reward

        if ep % 5 == 0: # 每隔10个episode输出一次平均reward
            print('episode', ep, ':', np.mean(all_scores[:-10:-1]), 'average score')

        if np.mean(all_scores[:-10:-1]) >= agent.goal: # 如果连续10个episode的平均reward大于goal，就保存模型
            successful_sequences += 1
            if successful_sequences == 5:
                print('success at episode', ep)
                # agent.save()
                return all_scores
        else:
            successful_sequences = 0 # 如果连续10个episode的平均reward小于goal，就重置successful_sequences

    return all_scores

def plot(scores):
    plt.plot(scores)
    plt.xlabel('Episode')
    plt.ylabel('Score')
    # plt.show()
    plt.savefig('./result/DDPG_scores.png')

def test(agent,episodes = 50,render = False):
    print('----------Start testing----------')
    scores = []
    for _ in range(episodes): # 测试50个episode
        state = agent.env.reset() # 初始化环境
        state = torch.tensor(state).to(agent.device).float().unsqueeze(0) # 将state转换为tensor
        episode_reward = 0 # 记录每个episode的reward
        done = False # 记录训练是否结束

        while not done:
            if render: # 如果render为True，则渲染环境
                agent.env.render()
            action = agent.act(state, 0) # 选择动作，epsilon为0，即完全按照actor网络选择动作
            state, reward, done, _ = agent.env.step(action) # 执行动作，返回下一个状态、reward、是否结束、以及info
            state = torch.tensor(state).to(agent.device).float().unsqueeze(0)
            episode_reward += reward

        scores.append(episode_reward)

    print('average score:', sum(scores) / len(scores))
    print('max reward:', max(scores))
    print('min reward:', min(scores))

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make('MountainCarContinuous-v0')
    n_actions = env.action_space.shape[0]  # 动作空间的维度，1
    n_states = env.observation_space.shape[0]  # 状态空间的维度，2
    Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))  # 定义Transition元组

    layers = (256, 256, n_states, n_actions)  # 定义神经网络的结构

    Model = DDPG(layers, polyak=0.999, \
                 critic_lr=0.005, critic_optim_method=optim.Adam, critic_loss=F.mse_loss, \
                 actor_lr=0.0005, actor_optim_method=optim.Adam)  # 定义DDPG模型

    noise = Noise(0, 0.15, 0.2, n_actions)  # 定义噪声

    MountainCarAgent = Agent(device, Transition, env, Model, noise, goal=91, min_score=-100, \
                             gamma=0.9, batch_size=128, memory_size=20000, max_episode=100, upd_rate=1,
                             exploration_episodes=10)  # 定义agent

    scores = train(MountainCarAgent)
    plot(scores)
    test(MountainCarAgent, episodes=10, render=True)
