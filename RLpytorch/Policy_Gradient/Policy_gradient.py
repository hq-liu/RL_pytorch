from torch import nn
from torch.nn import functional as F
import numpy as np
from torch.autograd import Variable
import torch
from torch import optim
from torch.nn import init


class policy_gradient_net(nn.Module):
    def __init__(self, state_dim, action_dim, num_hidden=3, hidden_size=128):
        """
        Policy gradient network
        :param state_dim: 状态数
        :param action_dim: 动作数
        :param hidden_size: 隐层节点数
        """
        super(policy_gradient_net, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.input_layer = nn.Linear(in_features=self.state_dim,
                                     out_features=self.hidden_size)
        init.normal(self.input_layer.weight, std=0.3, mean=0)
        init.constant(self.input_layer.bias, 0.1)
        self.hidden_layers = nn.ModuleList([nn.Linear(in_features=hidden_size,
                                                     out_features=hidden_size) for _ in range(num_hidden)])
        for i in self.hidden_layers:
            init.normal(i.weight, std=0.3, mean=0)
            init.constant(i.bias, 0.1)
        self.output_layer = nn.Linear(in_features=self.hidden_size,
                                      out_features=self.action_dim)
        init.normal(self.output_layer.weight, std=0.3, mean=0)
        init.constant(self.output_layer.bias, 0.1)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        for i in self.hidden_layers:
            x = F.relu(i(x))
        x = self.output_layer(x)
        return F.softmax(x)


class policy_gradient_learning():
    def __init__(self, state_dim, action_dim, learning_rate=0.02, reward_decay=0.99, use_gpu=False):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = learning_rate
        self.gamma = reward_decay
        self.use_gpu = use_gpu

        self.FloatTensor = torch.cuda.FloatTensor if use_gpu else torch.FloatTensor
        self.LongTensor = torch.cuda.LongTensor if use_gpu else torch.LongTensor
        self.ByteTensor = torch.cuda.ByteTensor if use_gpu else torch.ByteTensor
        # TODO 这边将神经网络规模减小可以正确训练，原因不明（可能是过拟合）
        self.net = policy_gradient_net(state_dim, action_dim, 1, 20).type(self.FloatTensor)
        self.optimizer = optim.Adam(self.net.parameters(),lr=self.lr)
        # 经历
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []

    def choose_action(self, observation):
        observation = Variable(torch.FloatTensor(observation)).type(self.FloatTensor)
        action_probility = self.net(observation)  # 输出各个动作的概率值
        action_probility = action_probility.data.cpu().numpy()
        action = np.random.choice(range(self.action_dim), p=action_probility.ravel())
        return action

    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def _discount_and_norm_rewards(self):
        # discount episode rewards
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        # normalize episode rewards
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs

    def learn(self):
        discounted_ep_rs_norm = self._discount_and_norm_rewards()
        obs = np.vstack(self.ep_obs)
        acts = np.array(self.ep_as)
        vt = discounted_ep_rs_norm
        obs, acts, vt = Variable(torch.FloatTensor(obs)).type(self.FloatTensor), \
                        Variable(torch.LongTensor(acts)).type(self.LongTensor), \
                        Variable(torch.FloatTensor(vt)).type(self.FloatTensor)

        # 这边的loss没有问题
        action_probability = self.net(obs).gather(1, acts.view(-1, 1))
        log_prob = -torch.log(action_probability)
        vt = vt.view_as(log_prob)
        loss = torch.mean(log_prob * vt)


        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []
        torch.cuda.empty_cache()
        return discounted_ep_rs_norm

    def store_params(self):
        torch.save(self.net.state_dict(), 'pg.pkl')

    def load_params(self):
        self.net.load_state_dict(torch.load('pg.pkl'))

