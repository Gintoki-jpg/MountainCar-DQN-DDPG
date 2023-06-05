from Algorithm import ValueNetwork, PolicyNetwork
import torch
import torch.optim as optim
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")

class DDPG:
    def __init__(self, layers_sizes, polyak=0.9999,
                 critic_lr=0.0001, critic_optim_method=optim.Adam, critic_loss=F.mse_loss,
                 actor_lr=0.0001, actor_optim_method=optim.Adam):
        self.polyak = polyak # 软更新的系数

        self.CriticEstimate = ValueNetwork(*layers_sizes) # 创建Critic的估计网络
        self.CriticTarget = ValueNetwork(*layers_sizes)  # 创建Critic的目标网络
        self.CriticTarget.load_state_dict(self.CriticEstimate.state_dict()) # 将估计网络的参数复制给目标网络
        self.critic_loss = critic_loss # Critic网络的损失函数
        self.critic_optimizer = critic_optim_method(self.CriticEstimate.parameters(), lr=critic_lr) # Critic网络的优化器

        self.ActorEstimate = PolicyNetwork(*layers_sizes) # 创建Actor的估计网络
        self.ActorTarget = PolicyNetwork(*layers_sizes) # 创建Actor的目标网络
        self.ActorTarget.load_state_dict(self.ActorEstimate.state_dict()) # 将估计网络的参数复制给目标网络
        self.actor_optimizer = actor_optim_method(self.ActorEstimate.parameters(), lr=actor_lr) # Actor网络的优化器

    def Q_estimate(self, state, action): # Critic网络的估计值
        return self.CriticEstimate(state, action)

    def Q_target(self, state, action): # Critic网络的目标值
        return self.CriticTarget(state, action)

    def action_estimate(self, state): # Actor网络的估计值
        return self.ActorEstimate(state)

    def action_target(self, state): # Actor网络的目标值
        return self.ActorTarget(state)

    def update_critic_params(self, estimates, targets): # 更新Critic网络的参数
        loss = self.critic_loss(estimates, targets) # 计算损失
        self.critic_optimizer.zero_grad() # 梯度清零
        loss.backward() # 反向传播
        torch.nn.utils.clip_grad_norm_(self.CriticEstimate.parameters(), 1) # 梯度裁剪
        self.critic_optimizer.step() # 更新参数

    def update_actor_params(self, states):
        loss = self.actor_loss(states)
        self.actor_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.ActorEstimate.parameters(), 1)
        self.actor_optimizer.step()

    def actor_loss(self, states): # Actor网络的损失函数
        actions = self.action_estimate(states) # 估计值
        return -self.Q_estimate(states, actions).mean() # 最大化Q值

    def update_target_networks(self):
        self.soft_update(self.ActorEstimate, self.ActorTarget) # 软更新Actor网络
        self.soft_update(self.CriticEstimate, self.CriticTarget) # 软更新Critic网络

    def soft_update(self, estimate_model, target_model): # 软更新
        for estimate_param, target_param in zip(estimate_model.parameters(), target_model.parameters()): # 更新每一个参数
            target_param.data.copy_(target_param.data * self.polyak + estimate_param.data * (1 - self.polyak))






