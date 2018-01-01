from torch import nn
from torch.nn import functional as F
import numpy as np
from torch.autograd import Variable
import torch
from torch import optim
from torch.nn import init


class networks(nn.Module):
    def __init__(self, in_dim, out_dim, num_hidden, hidden_size):
        super(networks, self).__init__()
        self.input = nn.Linear(in_features=in_dim, out_features=hidden_size)
        init.normal(self.input.weight, mean=0, std=0.1)
        init.constant(self.input.bias, 0.1)
        self.hidden_layers = nn.ModuleList([nn.Linear(in_features=hidden_size,
                                                      out_features=hidden_size) for _ in range(num_hidden)])
        for i in self.hidden_layers:
            init.normal(i.weight, mean=0, std=0.1)
            init.constant(i.bias, 0.1)
        self.output = nn.Linear(in_features=hidden_size, out_features=out_dim)
        init.normal(self.output.weight, mean=0, std=0.1)
        init.constant(self.output.bias, 0.1)

    def forward(self, x):
        x = F.relu(self.input(x))
        for i in self.hidden_layers:
            x = F.relu(i(x))
        x = self.output(x)
        return x

class Actor():
    def __init__(self, state_dim, action_dim, learning_rate=1e-3, use_gpu=False):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = learning_rate

        self.FloatTensor = torch.cuda.FloatTensor if use_gpu else torch.FloatTensor
        self.LongTensor = torch.cuda.LongTensor if use_gpu else torch.LongTensor
        self.ByteTensor = torch.cuda.ByteTensor if use_gpu else torch.ByteTensor

        self.net = networks(state_dim, action_dim, 1, 20).type(self.FloatTensor)
        self.optimizer = optim.Adam(self.net.parameters(),lr=self.lr)

    def learn(self, s, a, td_error):
        a = int(a)
        s = Variable(torch.FloatTensor(s)).type(self.FloatTensor)
        action_prob = F.softmax(self.net(s))[a]
        log_prob = torch.log(action_prob)
        exp_v = torch.mean(log_prob * td_error)
        loss_a = -exp_v
        self.optimizer.zero_grad()
        loss_a.backward(retain_graph=True)
        self.optimizer.step()
        torch.cuda.empty_cache()

    def choose_action(self, observation):
        observation = Variable(torch.FloatTensor(observation)).type(self.FloatTensor)
        action_probility = self.net(observation)  # 输出各个动作的概率值
        action_probility = F.softmax(action_probility)
        action_probility = action_probility.data.cpu().numpy()
        action = np.random.choice(range(self.action_dim), p=action_probility.ravel())
        # action = np.argmax(action_probility)
        return action


class Critic():
    def __init__(self, state_dim, learning_rate=1e-2, use_gpu=False, reward_decay=0.9):
        self.state_dim = state_dim
        self.lr = learning_rate
        self.gamma = reward_decay

        self.FloatTensor = torch.cuda.FloatTensor if use_gpu else torch.FloatTensor
        self.LongTensor = torch.cuda.LongTensor if use_gpu else torch.LongTensor
        self.ByteTensor = torch.cuda.ByteTensor if use_gpu else torch.ByteTensor

        self.net = networks(state_dim, 1, 1, 20).type(self.FloatTensor)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)

    def learn(self, s, r, s_):

        s, r, s_ = Variable(torch.from_numpy(s)).type(self.FloatTensor), \
                   Variable(torch.FloatTensor([r])).type(self.FloatTensor), \
                   Variable(torch.from_numpy(s_)).type(self.FloatTensor)
        v_ = self.net(s_)
        v = self.net(s)
        td_error = r + self.gamma * v_ - v
        loss_c = td_error ** 2
        self.optimizer.zero_grad()
        loss_c.backward(retain_graph=True)
        self.optimizer.step()
        torch.cuda.empty_cache()
        return td_error

