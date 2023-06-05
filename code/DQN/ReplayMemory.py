import random
from collections import  deque
import warnings
warnings.filterwarnings("ignore")

class ReplayMemory(object): # 经验回放算法
    def __init__(self, capacity,transition):
        self.capacity = capacity # 经验池容量
        self.memory = deque(maxlen=capacity)  # 使用deque数据结构作为经验池，设置最大长度为容量
        self.Transition = transition

    def push(self, *args):
        self.memory.append(self.Transition(*args))  # 直接使用append方法添加经验

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size) # 从经验池中随机采样
        return batch

    def __len__(self):
        return len(self.memory) # 返回经验池中的经验数量
