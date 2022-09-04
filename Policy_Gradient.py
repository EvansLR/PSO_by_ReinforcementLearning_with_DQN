"""
This part of code is the reinforcement learning brain, which is a brain of the agent.
All decisions are made in here.

Policy Gradient, Reinforcement Learning.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

"""

import numpy as np
import tensorflow as tf

# reproducible
np.random.seed(2)
tf.set_random_seed(1)


class PolicyGradient:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.95,
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


        # 执行初始化计算图 并初始化所有参数
        self.sess = tf.Session()

        self.sess.run(tf.global_variables_initializer())


    # 构建网络函数
    def _build_net(self):
        # 首先构建输入层内容
        with tf.name_scope('inputs'):
            # 接收observation的值
            self.tf_obs = tf.placeholder(tf.float32, [None, self.n_features], name="observations")
            # 接收我们在这个回合选择过的actions
            self.tf_acts = tf.placeholder(tf.int32, [None, ], name="actions_num")
            # 衡量某个state-action对应的value值
            self.tf_vt = tf.placeholder(tf.float32, [None, ], name="actions_value")

        # FC1：全连接层1
        layer = tf.layers.dense(
            inputs=self.tf_obs,
            # 输出个数
            units=10,
            #激活函数
            activation=tf.nn.tanh,
            #权重初始化正态分布 均值为0，标准差为0.3
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            #偏置项常量初始化
            bias_initializer=tf.constant_initializer(0.1),
            # 全连接层第一层的名字
            name='fc1'
        )
        # FC2：全连接层2
        all_act = tf.layers.dense(
            # 输入为全连接层的第一层
            inputs=layer,
            # 输出个数为动作的数目
            units=self.n_actions,
            # 暂时不加入激活函数
            activation=None,
            # 初始化权重系数
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),

            bias_initializer=tf.constant_initializer(0.1),
            # 全连接层2的名字
            name='fc2'
        )

        # softmax函数得到每个动作的概率值
        self.all_act_prob = tf.nn.softmax(all_act, name='act_prob')  # use softmax to convert to probability



        # 定义损失函数：交叉熵损失
        with tf.name_scope('loss'):
            #
            neg_log_prob = tf.reduce_sum(-tf.log(self.all_act_prob)*tf.one_hot(self.tf_acts, self.n_actions), axis=1)
            loss = tf.reduce_mean(neg_log_prob * self.tf_vt)  # reward guided loss


        # 采用tensorflow的内置优化方法，最小化loss
        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

    # 选择action
    # 根据action的概率选择
    def choose_action(self, observation):
        # 所有action的概率
        prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.tf_obs: observation[np.newaxis, :]})
        # 根据概率选择action p拉成一维数组
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())
        # 返回选择的动作
        return action


    # 存储一个episode里的所有动作，每一个回合结束都需要清空
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
