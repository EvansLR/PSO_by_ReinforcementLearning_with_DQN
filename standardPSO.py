import numpy as np
from CEC2006 import  G01,G02,G03,G04,G05



# 粒子类
class Partical:
    # 进行粒子的初始化
    # 参数分别为 自变量取值范围中的最小值和最大值，最大速度和最小速度和适应值函数
    def __init__(self, x_min, x_max, max_v, min_v, Problem:G01):
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
        self.punish=Problem.punish

        # 计算当前位置的适应值
        self.bestFitness = self.fitness(self.pos)
        # 计算当前位置的惩罚值
        self.best_punish=self.punish(self.pos)



    # 粒子适应值更新函数:
    def _updateFit(self):


        # 假如单个粒子 当前位置和历史上最好的都没有违反约束
        if  self.punish(self.pos)<= 0.0001 and self.best_punish <= 0.0001:
            if self.fitness(self.pos) < self.bestFitness:
                self.bestFitness = self.fitness(self.pos)
                self.pbest=self.pos
                self.best_punish=self.punish(self.pos)


        # 假如当前没有违反，而历史违反了，则取当前的位置
        elif  self.punish(self.pos)<= 0.0001 and self.best_punish > 0.0001:
            self.bestFitness = self.fitness(self.pos)
            self.pbest = self.pos
            self.best_punish = self.punish(self.pos)


        # 两个都违反，则取适应度小的
        elif self.punish(self.pos)> 0.0001 and self.best_punish > 0.0001:
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
class PSO_Standard:
    # 类的初始化
    # pop参数代表种群大小 generation代表代数 x_min x_max代表自变量取值范围  fitnessFunction为适应值函数
    # c1代表个体认知常数 c2代表社会经验常数
    # w一般为1，代表这是一个常规的PSO算法
    def __init__(self, pop_size, generation,m_FUNC:G03, c1=2.05, c2=2.05, w=1):
        # 初始化c1 c2 和w
        self.c1 = c1
        self.c2 = c2
        self.w = w  # 惯性因子

        # 惯性因子衰减系数
        self.pop = pop_size  # 种群大小
        # 初始化变量的取值范围
        self.x_min = np.array(m_FUNC.lower_bounds)
        self.x_max = np.array(m_FUNC.upper_bounds)
        # 初始化迭代次数
        self.generation = generation
        # 初始化最大速度，一般为每维变量的变化范围的
        self.max_v = (self.x_max - self.x_min) * 0.30
        self.min_v = -(self.x_max - self.x_min) * 0.30
        # 记录函数
        self.fitnessFunction = m_FUNC
        # 初始化种群
        self.particals = [Partical(self.x_min, self.x_max, self.max_v, self.min_v, self.fitnessFunction) for i in range(self.pop)]

        # 获得全局最佳的信息
        self.gbest = np.zeros(len(self.x_min))
        # 目前的适应值设置为无穷大
        self.gbestFit = float('Inf')
        # 保存全局最好的惩罚值
        self.gbestPunish=float('Inf')

        # 保存每次的最佳适应值
        self.fitness_list = []
        # 保存每次最佳适应值所在的位置
        self.best_pos=[]

    def init_gbest(self):
        for part in self.particals:
            if part.getBestFit() < self.gbestFit:
                self.gbestFit = part.getBestFit()
                self.gbest = part.getPbest()
                self.gbestPunish=part.best_punish




    def done(self):
        self.init_gbest()
        self.fitnessFunction.generation=0
        # 根据迭代次数进行循环遍历
        for i in range(self.generation):
            self.fitnessFunction.generation += 1
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
                        #更新全局最优惩罚
                        self.gbestPunish=part.best_punish
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

            # print(self.gbestFit)
            self.fitness_list.append(self.gbestFit)
            self.best_pos.append(self.gbest)
        # 返回每次迭代最优值的位置以及最终的最优值
        self.fitnessFunction.generation = 0
        return self.fitness_list

#
# if __name__=="__main__":
#     pop_size=500
#     generation=2000
#     func=G02()
#
#     problem=PSO_Standard(pop_size,generation,func)
#
#     problem.done()
#     print('最优位置',problem.gbest)
#     print('最优适应值',problem.fitnessFunction.fitness(problem.gbest))
#     print('约束情况',func.constraints(problem.gbest))
