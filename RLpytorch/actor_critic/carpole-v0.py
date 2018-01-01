import gym
from actor_critic.Actor_critic import *
import torch

env = gym.make('CartPole-v0')
env.seed(1)
np.random.seed(1)
torch.cuda.manual_seed(1)
env = env.unwrapped
use_cuda = torch.cuda.is_available()

actor = Actor(state_dim=env.observation_space.shape[0],
              action_dim=env.action_space.n, use_gpu=use_cuda)
critic = Critic(state_dim=env.observation_space.shape[0], use_gpu=use_cuda)

for i in range(3000):
    s = env.reset()
    track_r = []
    while True:
        a = actor.choose_action(observation=s)
        s_, r, done, _ = env.step(a)
        track_r.append(r)
        td_error = critic.learn(s, r, s_)
        actor.learn(s, a, td_error)

        if done:
            ep_rs_sum = sum(track_r)
            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.95 + ep_rs_sum * 0.05

            print("episode:", i, "  reward:", int(running_reward))
            break

        s = s_