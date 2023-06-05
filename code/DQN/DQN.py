import copy
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
warnings.filterwarnings("ignore")

class DQN:
    def __init__(self, layers, lr=0.0005, optim_method=optim.Adam):
        self.layers = layers # 神经网络层
        self.lr = lr # 学习率
        self.loss = nn.MSELoss()  # 使用nn.MSELoss作为损失函数
        self.optim_method = optim_method # 优化器
        self.TargetNetwork = None # 目标网络
        self.EstimateNetwork = None # 估计网络
        self.optimizer = None # 优化器
        self.build_model() # 构建模型

    def build_model(self): # 构建模型
        def init_weights(layer):
            if isinstance(layer, nn.Linear):  # 使用isinstance检查类型
                nn.init.xavier_normal_(layer.weight)

        self.EstimateNetwork = nn.Sequential(*self.layers) # 使用nn.Sequential构建神经网络
        self.EstimateNetwork.apply(init_weights) # 使用apply方法初始化权重

        self.TargetNetwork = copy.deepcopy(self.EstimateNetwork)  # 使用copy.deepcopy进行深拷贝
        self.TargetNetwork.load_state_dict(self.EstimateNetwork.state_dict()) # 使用load_state_dict加载参数

        self.optimizer = self.optim_method(self.EstimateNetwork.parameters(), lr=self.lr) # 使用优化器优化估计网络

    def Q_target(self, inp): # 计算目标值
        return self.TargetNetwork(inp)

    def Q_estimate(self, inp): # 计算估计值
        return self.EstimateNetwork(inp)

    def update_target(self): # 更新目标网络
        self.TargetNetwork.load_state_dict(self.EstimateNetwork.state_dict())

    def update_parameters(self, estimated, targets): # 更新参数
        loss = self.loss(estimated, targets.unsqueeze(1)) # 计算损失
        self.optimizer.zero_grad() # 梯度清零
        loss.backward() # 反向传播

        for param in self.EstimateNetwork.parameters(): # 使用梯度裁剪
            param.grad.data.clamp_(-1, 1) # 使用clamp_方法裁剪梯度
        self.optimizer.step() # 更新参数


