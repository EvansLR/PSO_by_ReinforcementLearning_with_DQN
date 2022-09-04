import gym.spaces
import numpy as np
from CEC2006 import G01, G02,G03,G04,G05
from gym import spaces
import  copy
# 设置随机数种子
np.random.seed(10)
gym.spaces.seed(5)


# 粒子类
class Partical:
    # 进行粒子的初始化
    # 参数分别为 自变量取值范围中的最小值和最大值，最大速度和最小速度和适应值函数
    def __init__(self, x_min, x_max, max_v, min_v, Problem: G01):
        # 获得变量的维度
        self.dim = len(x_min)
        # 获得变量的最大速度
        self.max_v = max_v
        # 获得变量的最小速度
        self.min_v = min_v
        # 获得变量取值范围的最大值
        self.x_min = x_min
        # 获得变量取值范围的最小值
        self.x_max = x_max
        # 记录变量的当前位置
        self.pos = np.zeros(self.dim)
        # 记录变量的最好位置
        self.pbest = np.zeros(self.dim)

        # 进行变量位置的初始化（初始位置）
        self.initPos(x_min, x_max)
        # 初始化当前的速度
        self._v = np.zeros(self.dim)
        # 进行速度初始化
        self.initV(min_v, max_v)

        # 保存适应值函数
        self.fitness = Problem.evalution
        # 保存惩罚值计算函数
        self.punish = Problem.punish

        # 计算当前位置的适应值
        self.bestFitness = self.fitness(self.pos)
        # 计算当前位置的惩罚值
        self.best_punish = self.punish(self.pos)


    # 粒子适应值更新函数:
    def _updateFit(self):
        # 假如适应值优化
        # if self.fitness(self.pos) < self.bestFitness:
        #     # 保存为现在的适应值
        #     self.bestFitness = self.fitness(self.pos)
        #     # 记录下当前的位置
        #     self.pbest = self.pos

        # 假如单个粒子 当前位置和历史上最好的都没有违反约束
        if self.punish(self.pos) <= 0.00001 and self.best_punish <= 0.00001:
            if self.fitness(self.pos) < self.bestFitness:
                self.bestFitness = self.fitness(self.pos)
                self.pbest = self.pos
                self.best_punish = self.punish(self.pos)


        # 假如当前没有违反，而历史违反了，则取当前的位置
        elif self.punish(self.pos) <= 0.00001 and self.best_punish > 0.00001:
            self.bestFitness = self.fitness(self.pos)
            self.pbest = self.pos
            self.best_punish = self.punish(self.pos)


        # 两个都违反，则取适应度小的
        elif self.punish(self.pos) > 0.00001 and self.best_punish > 0.00001:
            if self.fitness(self.pos) < self.bestFitness:
                self.bestFitness = self.fitness(self.pos)
                self.pbest = self.pos
                self.best_punish = self.punish(self.pos)

    # 粒子位置更新函数
    def _updatePos(self):
        # 位置加速度更新粒子的位置
        self.pos = self.pos + self._v
        # 对于每一个维度，防止越界的操作
        for i in range(self.dim):
            self.pos[i] = min(self.pos[i], self.x_max[i])
            self.pos[i] = max(self.pos[i], self.x_min[i])
        # 一定的几率 重置位置
        #     if np.random.random()<0.3:
        #        # self.pos[i] = np.random.uniform(self.x_min[i], self.x_max[i])
        #        self.pos[i]=np.random.random()*(self.x_max[i]-self.x_min[i])+self.x_min[i]

    # 更新个体的速度函数
    def _updateV(self, w, c1, c2, gbest):
        '''这里进行的都是数组的运算'''
        # 速度的更新公式
        self._v = w * self._v + c1 * np.random.random() * (self.pbest - self.pos) + c2 * np.random.random() * (
                gbest - self.pos)
        # 对于每一个维度防止越界
        for i in range(self.dim):
            self._v[i] = min(self._v[i], self.max_v[i])
            self._v[i] = max(self._v[i], self.min_v[i])
            # 一定的几率 重置位置
            # if np.random.random() < 0.3:
            #     # self.pos[i] = np.random.uniform(self.x_min[i], self.x_max[i])
            #     self._v[i] = np.random.random() * (self.max_v[i] - self.min_v[i]) + self.min_v[i]

    # 变量位置初始化函数，参数为变量的取值范围
    def initPos(self, x_min, x_max):
        # 对于每一个维度依次初始化
        for i in range(self.dim):
            # 正态初始化
            self.pos[i] = np.random.uniform(x_min[i], x_max[i])
            # 将最好的位置暂存为当前的位置
            self.pbest[i] = self.pos[i]

    # 速度初始化函数，参数为速度的取值范围
    def initV(self, min_v, max_v):
        # 对于每个维度依次遍历
        for i in range(self.dim):
            # 正态初始化每个维度的速度
            self._v[i] = np.random.uniform(min_v[i], max_v[i])

    # 获取种群的最优解
    def getPbest(self):
        return self.pbest

    def getBestFit(self):
        return self.bestFitness

    # 粒子更新函数，w，c1，c2均为算法参数gbest为当前种群的最优值所在位置
    def update(self, w, c1, c2, gbest):
        # 更新例子的速度
        self._updateV(w, c1, c2, gbest)
        # 更新粒子的位置
        self._updatePos()
        # 更新粒子的适应值
        self._updateFit()


# 粒子群算法类
class env_PSO:
    # 类的初始化
    # pop参数代表种群大小 generation代表代数 x_min x_max代表自变量取值范围  fitnessFunction为适应值函数
    # c1代表个体认知常数 c2代表社会经验常数
    # w一般为1，代表这是一个常规的PSO算法
    def __init__(self, pop_size, generation, m_FUNC: G03, c1=2.05, c2=2.05, w=1):
        # 初始化c1 c2
        self.c1 = c1
        self.c2 = c2
        # 惯性因子
        self.w = w

        # 惯性因子衰减系数
        self.pop = pop_size  # 种群大小
        # 初始化变量的取值范围
        self.x_min = np.array(m_FUNC.lower_bounds)
        self.x_max = np.array(m_FUNC.upper_bounds)
        # 初始化最大迭代次数
        self.generation = generation
        # 初始化当前迭代次数
        self.now_iter=0

        # 初始化最大速度，一般为每维变量的变化范围的
        self.max_v = (self.x_max - self.x_min) * 0.3
        self.min_v = -(self.x_max - self.x_min) * 0.3
        # 记录函数
        self.fitnessFunction = m_FUNC
        # 初始化种群
        self.particals = [Partical(self.x_min, self.x_max, self.max_v, self.min_v, self.fitnessFunction) for i in
                          range(self.pop)]
        # 重新备份种群
        self.part_backup=copy.deepcopy(self.particals)

        # 获得全局最佳的信息
        self.gbest = np.zeros(len(self.x_min))
        # 目前的适应值设置为无穷大
        self.gbestFit = float('Inf')
        # 保存全局最好的惩罚值
        self.gbestPunish = float('Inf')

        # 保存每次的最佳适应值
        self.fitness_list = []
        # 保存每次最佳适应值所在的位置
        self.best_pos = []
        # 保存动作空间
        self.action_space = spaces.Discrete(7)
        # 初始化全局信息
        self.init_gbest()





    def reset(self,c1,c2,w):
    #     重置c1 c2 w
        # 初始化c1 c2
        self.c1 = c1
        self.c2 = c2
        # 惯性因子
        self.w = w

        # 重置迭代次数
        self.now_iter = 0

        # 重置函数内的迭代次数
        self.fitnessFunction.generation=0

        # 恢复最初的种群 深拷贝
        self.particals =copy.deepcopy(self.part_backup)


        # 获得全局最佳的信息
        self.gbest = np.zeros(len(self.x_min))
        # 目前的适应值设置为无穷大
        self.gbestFit = float('Inf')
        # 保存全局最好的惩罚值
        self.gbestPunish = float('Inf')

        # 保存每次的最佳适应值
        self.fitness_list = []
        # 保存每次最佳适应值所在的位置
        self.best_pos = []

        # 初始化全局信息
        self.init_gbest()

        # 返回当前的信息
        done,state = self.get_now_state()

        return state


    def init_gbest(self):
        for part in self.particals:
            if part.getBestFit() < self.gbestFit:
                self.gbestFit = part.getBestFit()
                self.gbest = part.getPbest()
                self.gbestPunish = part.best_punish

        # self.fitness_list.append(self.gbestFit)
        # self.best_pos.append(self.gbest)


    # 将当前的适应值转为奖励值
    def fit2reward(self,value):

        if value<0:
            r=abs(value)+1.0
        else:
            r=1.0/(value+1.0)

        return r


    # 单次循环更新粒子信息，并根据更新情况，返回当前一步的reward
    def update_part(self):
        reward=0
        if_change=False
        last_best=self.gbestFit
        # 记录此循环粒子适应度
        now_gen_best=float('inf')
        for part in self.particals:
            # 进行粒子的更新
            part.update(self.w, self.c1, self.c2, self.gbest)
            if part.getBestFit()<now_gen_best:
                now_gen_best=part.getBestFit()
            # 1 如果该粒子和历史最优均未违反约束
            if part.best_punish <= 0.0001 and self.gbestPunish <= 0.0001:
                if part.getBestFit() < self.gbestFit:
                    # 计算提升了多少，百分比
                    accu=abs(part.getBestFit()-self.gbestFit)/self.gbestFit
                    # 更新全局最优适应值
                    self.gbestFit = part.getBestFit()
                    # 更新全局最优适应值所在的位置
                    self.gbest = part.getPbest()
                    # 更新全局最优惩罚
                    self.gbestPunish = part.best_punish
                    # if accu>0.0005:
                    if_change=True


            # 2 如果历史违反了 当前未违反
            elif part.best_punish <= 0.0001 and self.gbestPunish > 0.0001:
                # 更新全局最优适应值
                self.gbestFit = part.getBestFit()
                # 更新全局最优适应值所在的位置
                self.gbest = part.getPbest()
                # 更新全局最优惩罚
                self.gbestPunish = part.best_punish

                # # 是否改变违反程度
                if_change=True

            # 3 如果都违反了
            elif part.best_punish > 0.0001 and self.gbestPunish > 0.0001:
                if part.getBestFit() < self.gbestFit:
                    accu = abs(part.getBestFit() - self.gbestFit) / self.gbestFit
                    self.gbestFit = part.getBestFit()
                    # 更新全局最优适应值所在的位置
                    self.gbest = part.getPbest()
                    # 更新全局最优惩罚
                    self.gbestPunish = part.best_punish


                    # if accu>0.0005:
                        # 是否历史最优改变
                    # if_change=True

        # 保存历史适应值变化情况
        self.fitness_list.append(self.gbestFit)
        self.best_pos.append(self.gbest)



        # 历史最优改变了才有奖励
        # 记录变化量作为奖励值
        # reward = self.fit2reward(self.gbestFit)-self.fit2reward(last_best)
        if self.now_iter==self.generation-1:
            reward = self.fit2reward(self.gbestFit)
        # #     print(reward)

        # if self.gbestFit<last_best or if_change:
        #     reward=self.fit2reward(self.gbestFit)
        # elif now_gen_best==last_best:
        #     reward=0.0
        # else:
        #     reward=-1.0

        return reward

    # 获取当前的状态
    def get_now_state(self):
        done=False
        # 判断该episode是否结束
        if self.now_iter==self.generation-1:
            done=True
        #衡量粒子之间的方差
        all_pos=[]
        for part in self.particals:
            all_pos.append(part.pos)
        all_pos=np.array(all_pos)
        # d1 代表粒子之间的方差
        d1=np.var(all_pos)
        # d2 代表当前的迭代次数所占比例
        d2=self.now_iter/self.generation
        # d3 代表过去5次适应值的平均变化
        d3=0
        for i in range(max(1,self.now_iter-5),self.now_iter):
            d3+=(self.fitness_list[i]-self.fitness_list[i-1])**2

        state=np.array([d1,d2,d3])

        return done,state






    # 标准PSO中每步迭代，不需要记录任何信息
    def done(self):

        # 根据迭代次数进行循环遍历
        for i in range(self.generation):
            # 对于每个粒子依次遍历
            for part in self.particals:

                # 进行粒子的更新
                part.update(self.w, self.c1, self.c2, self.gbest)
                # 进行全局最优位置的更新

                # 1 如果该粒子和历史最优均未违反约束
                if part.best_punish <= 0.0001 and self.gbestPunish <= 0.0001:
                    if part.getBestFit() < self.gbestFit:
                        # 更新全局最优适应值
                        self.gbestFit = part.getBestFit()
                        # 更新全局最优适应值所在的位置
                        self.gbest = part.getPbest()
                        # 更新全局最优惩罚
                        self.gbestPunish = part.best_punish
                # 2 如果历史违反了 当前未违反
                elif part.best_punish <= 0.0001 and self.gbestPunish > 0.0001:
                    # 更新全局最优适应值
                    self.gbestFit = part.getBestFit()
                    # 更新全局最优适应值所在的位置
                    self.gbest = part.getPbest()
                    # 更新全局最优惩罚
                    self.gbestPunish = part.best_punish
                # 3 如果都违反了
                elif part.best_punish > 0.0001 and self.gbestPunish > 0.0001:
                    if part.getBestFit() < self.gbestFit:
                        self.gbestFit = part.getBestFit()
                        # 更新全局最优适应值所在的位置
                        self.gbest = part.getPbest()
                        # 更新全局最优惩罚
                        self.gbestPunish = part.best_punish
            print(i)
            print(self.gbestFit)
            self.fitness_list.append(self.gbestFit)
            self.best_pos.append(self.gbest)
        # 返回每次迭代最优值的位置以及最终的最优值
        return self.fitness_list

    #基于强化学习的PSO的迭代过程
    def step(self,action):
    #根据action选择相应的变化
        # if action==0:
        #     self.c1+=0.01
        # elif action==1:
        #     self.c1-=0.01
        # elif action==2:
        #     self.c2+=0.01
        # elif action==3:
        #     self.c2-=0.01
        # elif action==4:
        #     self.w+=0.01
        # elif action==5:
        #     self.w-=0.01
        if action == 0:
            self.c1/=1.015
        elif action == 1:
            self.c1*=1.015
        elif action == 2:
            self.c2/=1.015
        elif action == 3:
            self.c2*=1.015
        elif action == 4:
            self.w *=1.015
        elif action == 5:
            self.w/=1.015
        elif action==6:
            # 不做任何动作
            self.w=self.w

        # 不断更改约束
        self.fitnessFunction.generation+=1

        # 更新粒子的信息 并返回当前该步的奖励
        reward=self.update_part()

        # 获取当前是否完成以及当前的状态
        done,state=self.get_now_state()


        # 更新迭代次数
        self.now_iter+=1


        # 这里不设置信息
        info=None
        return np.array(state),reward,done,info



