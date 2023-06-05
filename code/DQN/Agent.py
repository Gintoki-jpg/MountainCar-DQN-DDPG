import random
import torch
import warnings
warnings.filterwarnings("ignore")

from ReplayMemory import ReplayMemory

class Agent:
    def __init__(self, device, transition, env, model, n_actions, goal, min_score,
                 eps_start=1, eps_end=0.001, eps_decay=0.9, gamma=0.99,
                 batch_size=64, memory_size=100000, max_episode=2000, upd_rate=1):
        self.device = device # cpu or gpu
        self.Transition = transition
        self.env = env  # 环境
        self.n_actions = n_actions # 动作空间
        self.goal = goal # 目标分数
        self.min_score = min_score # 最低分数
        self.eps_start = eps_start # 初始epsilon
        self.eps = eps_start
        self.eps_end = eps_end # 最终epsilon
        self.eps_decay = eps_decay # epsilon衰减率
        self.gamma = gamma # 折扣因子
        self.batch_size = batch_size # 批大小
        self.target_update_rate = upd_rate # 目标网络更新频率
        self.model = model
        self.max_episode = max_episode # 最大训练轮数
        self.memory = ReplayMemory(memory_size,transition) # 经验回放池

    def act(self, state, eps):
        if random.random() < eps: # 若随机数小于epsilon，随机选择动作
            return torch.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long) # 随机选择动作
        else: # 否则使用当前策略
            with torch.no_grad():
                result = self.model.Q_estimate(state).max(1)[1] # 选择最大的动作
                return result.view(1, 1)

    def optimize(self): # 优化模型
        if len(self.memory) < self.batch_size: # 若经验池中的经验数量小于批大小，不进行优化
            return

        transitions = self.memory.sample(self.batch_size) # 从经验池中采样
        batch = self.Transition(*zip(*transitions)) # 将经验转换为批

        next_state_batch = torch.cat(batch.next_state) # 将批中的下一个状态拼接为一个张量
        state_batch = torch.cat(batch.state) # 将批中的状态拼接为一个张量
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        done_batch = torch.cat(batch.done)

        estimate_value = self.model.Q_estimate(state_batch).gather(1, action_batch) # 估计值

        Q_value_next = torch.zeros(self.batch_size, device=self.device) # 目标值
        with torch.no_grad():
            Q_value_next[~done_batch] = self.model.Q_target(next_state_batch).max(1)[0].detach()[~done_batch] # 若未结束，使用目标网络计算目标值
        target_value = (Q_value_next * self.gamma) + reward_batch # 计算目标值

        self.model.update_parameters(estimate_value, target_value) # 更新参数


