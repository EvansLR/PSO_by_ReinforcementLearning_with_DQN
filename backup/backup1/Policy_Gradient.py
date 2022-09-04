"""
This part of code is the reinforcement learning brain, which is a brain of the agent.
All decisions are made in here.

Policy Gradient, Reinforcement Learning.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.8.0
"""

import numpy as np
import tensorflow as tf

# reproducible
np.random.seed(1)
tf.set_random_seed(1)


class PolicyGradient:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.95,
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay

        # 存储每个回合的observation
        self.ep_obs=[]
        # 存储每个回合的action
        self.ep_as=[]
        # 存储每个回合的reward
        self.ep_rs =[]

        # 建立policy策略神经网络
        self._build_net()

        self.sess = tf.Session()

        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def _build_net(self):
        with tf.name_scope('inputs'):
            # 接收observation的值
            self.tf_obs = tf.placeholder(tf.float32, [None, self.n_features], name="observations")
            # 接收我们在这个回合选择过的actions
            self.tf_acts = tf.placeholder(tf.int32, [None, ], name="actions_num")
            # 衡量某个state-action对应的value值
            self.tf_vt = tf.placeholder(tf.float32, [None, ], name="actions_value")

        # FC1
        layer = tf.layers.dense(
            inputs=self.tf_obs,
            # 输出个数
            units=10,
            #激活函数
            activation=tf.nn.tanh,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc1'
        )
        # FC2
        all_act = tf.layers.dense(
            inputs=layer,
            units=self.n_actions,
            # 先不用softmax后续再加入
            activation=None,
            #
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc2'
        )

        # softmax函数得到每个动作的概率值
        self.all_act_prob = tf.nn.softmax(all_act, name='act_prob')  # use softmax to convert to probability



        with tf.name_scope('loss'):
            # 最大化 reward (log_p * R)  即最小化 -(log_p * R)
            neg_log_prob = tf.reduce_sum(-tf.log(self.all_act_prob)*tf.one_hot(self.tf_acts, self.n_actions), axis=1)
            loss = tf.reduce_mean(neg_log_prob * self.tf_vt)  # reward guided loss

        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

    # 选择action
    # 根据action的概率选择
    def choose_action(self, observation):
        # 所有action的概率
        prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.tf_obs: observation[np.newaxis, :]})
        # 根据概率选择action
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())  # select action w.r.t the actions prob
        # 返回选择的动作

        return action


    # 存储一个回合，每一个回合结束都需要清空
    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    # 学习过程
    def learn(self):
        discounted_ep_rs_norm = self._discount_and_norm_rewards()

        #在每个episode上进行训练
        self.sess.run(self.train_op, feed_dict={
             self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
             self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
             self.tf_vt: discounted_ep_rs_norm,  # shape=[None, ]
        })

        # 清空
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []
        # 返回价值Vt
        return discounted_ep_rs_norm


    # reward衰减过程
    def _discount_and_norm_rewards(self):

        discounted_ep_rs = np.zeros_like(self.ep_rs)



        running_add = 0.0
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add *self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add
        return discounted_ep_rs
