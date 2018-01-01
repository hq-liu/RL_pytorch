import gym
from Policy_Gradient.Policy_gradient import *
import torch

env = gym.make('CartPole-v0')
env.seed(1)
np.random.seed(1)
torch.cuda.manual_seed(1)
env = env.unwrapped
use_cuda = torch.cuda.is_available()

RL = policy_gradient_learning(state_dim=env.observation_space.shape[0],
                              action_dim=env.action_space.n,
                              use_gpu=use_cuda)

def train():
    for i in range(3000):
        observation = env.reset()

        while True:
            # env.render()
            action = RL.choose_action(observation)

            observation_, reward, done, info = env.step(action)

            RL.store_transition(observation, action, reward)

            ep_rs_sum = sum(RL.ep_rs)
            if ep_rs_sum > 20000:  # 奖励值足够大，不再训练
                RL.store_params()
                done = True

            if done:

                if 'running_reward' not in globals():
                    running_reward = ep_rs_sum
                else:
                    running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
                print("episode:", i, "  reward:", int(running_reward))
                vt = RL.learn()
                break
            observation = observation_

def test():
    for i in range(500):
        observation = env.reset()
        while True:
            env.render()
            action = RL.choose_action(observation)

            observation_, reward, done, info = env.step(action)

            RL.store_transition(observation, action, reward)

            ep_rs_sum = sum(RL.ep_rs)

            if ep_rs_sum > 10000:
                print('test successed!')

            if done:
                break
            observation = observation_


train()
# test()
