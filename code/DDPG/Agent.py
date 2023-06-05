from Algorithm import ReplayMemory
import torch
import warnings
warnings.filterwarnings("ignore")



class Agent:
    def __init__(self, device, transition, env, Model, noise, goal, min_score,
                 gamma=0.99, batch_size=64, memory_size=100000, max_episode=2000, upd_rate=1, exploration_episodes=10):
        self.device = device
        self.Transition = transition

        self.env = env # 环境
        self.Model = Model # 神经网络
        self.noise = noise # 噪声
        self.goal = goal # 目标分数
        self.min_score = min_score # 最小分数

        self.gamma = gamma # 折扣因子
        self.batch_size = batch_size
        self.memory = ReplayMemory(memory_size, transition) # 经验池
        self.max_episode = max_episode # 最大episode数
        self.target_update_rate = upd_rate # target网络更新频率
        self.exploration_episodes = exploration_episodes # epsilon-greedy策略的episode因子


    def act(self, state, eps): # 选择动作
        with torch.no_grad(): # 关闭梯度
            action = self.Model.action_estimate(state) # 基于当前状态选择动作
            noise = torch.tensor(eps * self.noise.make_noise()).unsqueeze(0) # 增加噪声
            action += noise # 在动作上增加噪声
        return action.clamp_(self.env.action_space.low[0], self.env.action_space.high[0]) # 限制动作范围在环境的动作空间内

    def optimize(self): # 优化器
        if len(self.memory) < self.batch_size: # 经验池中的样本数量小于batch_size，不需要进行优化
            return

        transitions = self.memory.sample(self.batch_size) # 从经验池中采样
        batch = self.Transition(*zip(*transitions)) # 将样本转换为batch

        next_state_batch = torch.cat(batch.next_state)
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        done_batch = torch.cat(batch.done)

        estimates = self.Model.Q_estimate(state_batch, action_batch) # 网络的估计值
        Q_next = torch.zeros(self.batch_size, device=self.device).unsqueeze(1) # 下一个状态的Q值
        with torch.no_grad():
            next_actions = self.Model.action_target(next_state_batch) # 下一个状态的动作
            Q_next[~done_batch] = self.Model.Q_target(next_state_batch, next_actions)[~done_batch] # 下一个状态的Q值
        targets = reward_batch.unsqueeze(1) + self.gamma * Q_next # 目标值
        self.Model.update_critic_params(estimates, targets) # 更新Critic网络参数
        self.Model.update_actor_params(state_batch) # 更新Actor网络参数

