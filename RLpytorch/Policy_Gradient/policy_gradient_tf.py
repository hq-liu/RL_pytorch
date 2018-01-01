import tensorflow as tf
import numpy as np


class policy_gradient_tf():
    def __init__(self, state_dim, action_dim, learning_rate=0.02, reward_decay=0.99):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = learning_rate
        self.gamma = reward_decay

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []

        self._build_net()
        self.sess = tf.Session()

        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, observation):
        prob_weights = self.sess.run(self.act_prob, feed_dict={self.observations: observation[np.newaxis, :]})
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())  # select action w.r.t the actions prob
        return action

    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def learn(self):
        discounted_ep_rs_norm = self._discount_and_norm_rewards()

        self.sess.run(self.train_op, feed_dict={
            self.observations: np.vstack(self.ep_obs),
            self.actions: np.array(self.ep_as),
            self.rewards: discounted_ep_rs_norm
        })
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []  # empty episode data
        return discounted_ep_rs_norm

    def _build_net(self):
        with tf.variable_scope('inputs'):
            self.observations = tf.placeholder(tf.float32, shape=[None, self.state_dim], name='observations')
            self.actions = tf.placeholder(tf.int32, shape=[None,], name='actions')
            self.rewards = tf.placeholder(tf.float32, shape=[None,], name='rewards')
        w_init = tf.random_normal_initializer(stddev=0.3, mean=0, dtype=tf.float32)
        b_init = tf.constant_initializer(0.1, dtype=tf.float32)
        with tf.variable_scope('policy_gradient_net'):
            fc1 = tf.layers.dense(
                inputs=self.observations, units=50, activation=tf.nn.relu,
                kernel_initializer=w_init, bias_initializer=b_init
            )
            fc2 = tf.layers.dense(
                inputs=fc1, units=50, activation=tf.nn.relu,
                kernel_initializer=w_init, bias_initializer=b_init
            )
            acts = tf.layers.dense(
                inputs=fc2, units=self.action_dim, activation=None,
                kernel_initializer=w_init, bias_initializer=b_init
            )

        self.act_prob = tf.nn.softmax(acts, name='action_probability')

        with tf.variable_scope('loss'):
            self.neg_log_prob = tf.reduce_sum(-tf.log(self.act_prob) * tf.one_hot(self.actions, self.action_dim),
                                              axis=1)
            self.loss = tf.reduce_mean(self.neg_log_prob * self.rewards)  # reward guided loss
            self.loss_ = -tf.log(tf.reduce_sum(self.act_prob * tf.one_hot(self.action_dim, self.action_dim), axis=1))

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

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

import gym
np.random.seed(1)
tf.set_random_seed(1)

env = gym.make('CartPole-v0')
env.seed(1)
env = env.unwrapped


RL = policy_gradient_tf(state_dim=env.observation_space.shape[0],
                              action_dim=env.action_space.n,)

def train():
    for i in range(3000):
        observation = env.reset()
        while True:
            # env.render()
            action = RL.choose_action(observation)

            observation_, reward, done, info = env.step(action)

            RL.store_transition(observation, action, reward)

            if done:
                ep_rs_sum = sum(RL.ep_rs)
                if 'running_reward' not in globals():
                    running_reward = ep_rs_sum
                else:
                    running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
                print("episode:", i, "  reward:", int(running_reward))
                vt = RL.learn()
                break
            observation = observation_

train()

