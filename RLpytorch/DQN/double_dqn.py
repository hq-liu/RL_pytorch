from torch import nn
from torch.nn import functional as F
import numpy as np
from torch.autograd import Variable
import torch
from torch import optim


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
        for i in self.hidden_layers:
            x = F.relu(i(x))
        x = self.output(x)
        return x


class double_dqn():
    def __init__(self, state_dim, action_dim, learning_rate=1e-3, reward_decay=0.9,
                 e_greedy=0.9, replace_target_iter=300, memory_size=300, batch_size=64,
                 e_greedy_increment=None, use_gpu=False):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = learning_rate
        self.gamma = reward_decay
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.eps_increment = e_greedy_increment
        self.epsilon = e_greedy
        self.memory_counter = 0
        self.learning_step_counter = 0

        self.FloatTensor = torch.cuda.FloatTensor if use_gpu else torch.FloatTensor
        self.LongTensor = torch.cuda.LongTensor if use_gpu else torch.LongTensor
        self.ByteTensor = torch.cuda.ByteTensor if use_gpu else torch.ByteTensor

        # target_net: input: s_prime; output: next q
        self.target_net = deep_q_network(self.state_dim, self.action_dim, 3, 128).type(self.FloatTensor)
        # evaluation_net: input: s_prime; output: evaluation of q value
        self.eval_net = deep_q_network(self.state_dim, self.action_dim, 3, 128).type(self.FloatTensor)

        # memory: store [s, a, s_, r]
        self.memory = np.zeros((self.memory_size, self.state_dim * 2 + 2))
        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=self.lr)

    def choose_action(self, observation):
        """
        选择动作，此处包括e_greedy 和神经网络输出
        :param observation: 观测值
        :return: 动作值
        """
        observation = Variable(torch.FloatTensor(observation)).type(self.FloatTensor)
        if np.random.uniform() < self.epsilon:
            action_value = self.eval_net(observation)
            action = np.argmax(action_value.data.cpu().numpy())
        else:
            action = np.random.randint(0, self.action_dim)
        return int(action)
        pass

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1
        pass

    def learn(self):
        if self.learning_step_counter % self.replace_target_iter == 0:
            print('\ntarget_params_replaced\n')
            self.target_net.load_state_dict(self.eval_net.state_dict())

        if self.memory_counter > self.memory_size:
            indices = np.random.choice(self.memory_size, self.batch_size)
        else:
            indices = np.random.choice(self.memory_counter, self.batch_size)
        batch_memory = self.memory[indices, :]
        state = batch_memory[:, :self.state_dim]  # (batch_size, state_dim)
        action = batch_memory[:, self.state_dim].astype(int)  # (batch_size,)
        reward = batch_memory[:, self.state_dim + 1]  # (batch_size, )
        next_state = batch_memory[:, -self.state_dim:]  # 从memory中sample出state, action, reward, next_state

        state, action, reward, next_state = Variable(torch.FloatTensor(state)).type(self.FloatTensor), \
                                            Variable(torch.LongTensor(action)).type(self.LongTensor), \
                                            Variable(torch.FloatTensor(reward)).type(self.FloatTensor), \
                                            Variable(torch.FloatTensor(next_state), volatile=True).type(self.FloatTensor)

        q_next = self.target_net(next_state)
        next_action = self.eval_net(next_state)
        next_action_index = np.argmax(next_action.data.cpu().numpy(), axis=1)
        next_action_index = Variable(torch.LongTensor(next_action_index)).type(self.LongTensor)
        next_state_values = q_next.gather(1, next_action_index.view(-1, 1))
        next_state_values.volatile = False
        reward = reward.view_as(next_state_values)
        q_target = reward + self.gamma * next_state_values

        # print('1:',action)
        # print('2:',self.eval_net(state))
        state_action_values = self.eval_net(state).gather(1, action.view(-1, 1))
        loss = F.smooth_l1_loss(state_action_values, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.eval_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        self.epsilon = self.epsilon + self.eps_increment if self.epsilon < 1 else 1
        self.learning_step_counter += 1
        torch.cuda.empty_cache()  # 释放显存 0.3的功能

    def store_params(self):
        torch.save(self.eval_net.state_dict(), 'double_dqn.pkl')

    def load_params(self):
        self.eval_net.load_state_dict(torch.load('double_dqn.pkl'))
