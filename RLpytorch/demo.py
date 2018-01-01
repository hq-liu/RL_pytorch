from torch import nn
from torch.nn import functional as F
import numpy as np
from torch.autograd import Variable
import torch


class deep_q_network(nn.Module):
    def __init__(self, state_dim, action_dim, n_layers, hidden_size):
        """
        DQN的target-net和eval-net
        :param state_dim: 输入状态的shape
        :param action_dim: 输出动作的个数
        :param n_layers: 隐含层个数（不包括input和output）
        :param hidden_size: 隐含层的节点数
        """
        super(deep_q_network, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_layers= n_layers
        self.hidden_size = hidden_size

        # build the net
        self.input = nn.Linear(in_features=self.state_dim, out_features=self.hidden_size)  # 输入
        self.hidden_layers = nn.ModuleList([nn.Linear(in_features=hidden_size,
                                                      out_features=hidden_size) for i in range(n_layers)])
        self.output = nn.Linear(in_features=self.hidden_size, out_features=self.action_dim)  # 输出

    def forward(self, x):
        x = F.relu(self.input(x))
        x = F.relu(self.hidden_layers(x))
        x = self.output(x)
        return x


a = np.array([ 0.5308,
 0.5316,
 0.5327,
 0.5339,
 0.5345,
 0.4654,
 0.4656,
 0.5338,
 0.4656,
 0.4662,
 0.4670,
 0.5317,
 0.5331,
 0.4660,
 0.4667,
 0.5323,
 0.5335,])
b = np.array([ 0.5308,
 0.5316,
 0.5327,
 0.5339,
 0.5345,
 0.4654,
 0.4656,
 0.5338,
 0.4656,
 0.4662,
 0.4670,
 0.5317,
 0.5331,
 0.4660,
 0.4667,
 0.5323,
 0.5335,])
print(a.dot(b.T))
# a = np.array(range(15)).reshape((5,3))
# a = torch.from_numpy(a)
# a = Variable(a).type(torch.cuda.FloatTensor)
#
# b = model(a)
