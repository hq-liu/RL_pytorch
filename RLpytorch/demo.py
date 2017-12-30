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
        self.hidden_layers = {}  # 隐藏层
        # for i in range(self.n_layers):
        self.hidden_layers[1] = nn.Linear(in_features=self.hidden_size,
                                                   out_features=self.hidden_size)
        self.output = nn.Linear(in_features=self.hidden_size, out_features=self.action_dim)  # 输出

    def forward(self, x):
        x = F.relu(self.input(x))
        # for i in range(self.n_layers):
        x = F.relu(self.hidden_layers[1](x))
        x = self.output(x)
        return x


model = deep_q_network(3,2,3,50).cuda()
print(model)
# a = np.array(range(15)).reshape((5,3))
# a = torch.from_numpy(a)
# a = Variable(a).type(torch.cuda.FloatTensor)
#
# b = model(a)
