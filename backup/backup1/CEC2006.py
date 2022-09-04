# G01
import  numpy as np
class G01:
    # 初始化问题
    def __init__(self):
        # 函数的维度
        self.dimension=13
        # 函数的定义域下界
        self.lower_bounds = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        # 函数定义域上界
        self.upper_bounds = [1, 1, 1, 1, 1, 1, 1, 1, 1, 100, 100, 100, 1]
        # 保存函数的约束违反度
        self.g = np.empty(9)
        # 保存函数最优值所在的位置
        self.optimum=np.array([1,1,1,1,1,1,1,1,1,3,3,3,1])
        # 保存函数最优值的大小
        self.f_optimum=-15
        # 保存惩罚因子大小
        self.alpha=5000

        self.generation = 0


    # 计算函数值
    def fitness(self,x):
        return 5 * np.sum(x[0:4]) - 5 * np.sum(x[0:4] ** 2) - np.sum(x[4:13])

    # 保存约束条件
    def constraints(self,x):
        # 不等式约束
        self.g[0] =2 * x[0] + 2 * x[1] + x[9] + x[10] - 10
        self.g[1] = 2 * x[0] + 2 * x[2] + x[9] + x[11] - 10
        self.g[2] = 2 * x[1] + 2 * x[2] + x[10] + x[11] - 10
        self.g[3] = -8 * x[0] + x[9]
        self.g[4] = -8 * x[1] + x[10]
        self.g[5] = -8 * x[2] + x[11]
        self.g[6] = -2 * x[3] - x[4] + x[9]
        self.g[7] = -2 * x[5] - x[6] + x[10]
        self.g[8] = -2 * x[7] - x[8] + x[11]

        return self.g

    #求解约束违反度
    def penalty(self,g1):
        z=0
        # 均为不等式约束，可以直接是否大于0直接求约束违反度
        for i in range(len(g1)):
            if g1[i]>0:
                z+=(g1[i])
        return z

    def evalution(self,x1):

        #得到约束条件
        g1=self.constraints(x1)
        # 得到约束违反度
        p = self.penalty(g1)
        # 得到目标函数值
        fit_=self.fitness(x1)
        # 合并求和
        eval=fit_+(self.alpha*p)
        # 返回评估值
        return eval

    def punish(self,x1):

        #得到约束条件
        g1=self.constraints(x1)
        # 得到约束违反度
        p = self.penalty(g1)

        # 返回评估值
        return p



class G02:
    # The optimum solution is x* = (3.16246061572185, 3.12833142812967, 3.09479212988791, 3.06145059523469,
    #                               3.02792915885555, 2.99382606701730, 2.95866871765285, 2.92184227312450,
    #                               0.49482511456933, 0.48835711005490, 0.48231642711865, 0.47664475092742,
    #                               0.47129550835493, 0.46623099264167, 0.46142004984199, 0.45683664767217,
    #                               0.45245876903267, 0.44826762241853, 0.44424700958760, 0.44038285956317)
    # where f(x*) = -0.80361910412559.

    def __init__(self):
        self.dimension = 20
        self.lower_bounds = self.dimension * [0]
        self.upper_bounds = self.dimension * [10]
        self.g = np.empty(2)
        # 保存惩罚因子大小
        self.alpha=5000

        self.generation = 0

    def fitness(self, x):
        sum_jx = 0

        for j in range(self.dimension):
            sum_jx = sum_jx + (j + 1) * x[j]**2

        return -np.abs((np.sum(np.cos(x)**4) - 2 * np.prod(np.cos(x)**2))/np.sqrt(sum_jx))



    def constraints(self, x):
        self.g[0] = 0.75 - np.prod(x)
        self.g[1] = np.sum(x) - 7.5 * self.dimension

        return self.g

    def evalution(self,x1):

        #得到约束条件
        g1=self.constraints(x1)
        # 得到约束违反度
        p = self.penalty(g1)
        # 得到目标函数值
        fit_=self.fitness(x1)
        # 合并求和
        eval=fit_+(self.alpha*p)
        # 返回评估值
        return eval

    #求解约束违反度
    def penalty(self,g1):
        z=0
        # 均为不等式约束，可以直接是否大于0直接求约束违反度
        for i in range(len(g1)):
            if g1[i]>0:
                z+=(g1[i])
        return z


    def punish(self,x1):

        #得到约束条件
        g1=self.constraints(x1)
        # 得到约束违反度
        p = self.penalty(g1)
        # 返回评估值
        return p




# G03
class G03:
    # The optimum solution is x* = (0.31624357647283069, 0.316243577414338339, 0.316243578012345927,
    #                               0.316243575664017895, 0.316243578205526066, 0.31624357738855069,
    #                               0.316243575472949512, 0.316243577164883938, 0.316243578155920302,
    #                               0.316243576147374916)
    # where f(x*) = -1.00050010001000.

    def __init__(self):
        self.dimension = 10
        self.lower_bounds = self.dimension * [0]
        self.upper_bounds = self.dimension * [1]
        self.g = np.empty(2)

         #记录约束变化
        self.deltaIni = self.dimension *np.log10(max(np.array(self.upper_bounds) - np.array(self.lower_bounds)))
        # 保存惩罚因子大小
        self.alpha = 5000
    #     保存当前代数
        self.generation=0


    def fitness(self, x):

        return -(np.sqrt(self.dimension))**self.dimension * np.prod(x)

    def constraints(self, x):
        self.g[0] = abs(np.sum(x**2) - 1) -max(self.deltaIni/(1.015**self.generation),1e-4)

        return self.g

    def evalution(self, x1):

        # 得到约束条件
        g1 = self.constraints(x1)
        # 得到约束违反度
        p = self.penalty(g1)
        # 得到目标函数值
        fit_ = self.fitness(x1)
        # 合并求和
        eval = fit_ + (self.alpha*(self.generation+1) * p)
        # 返回评估值
        return eval

        # 求解约束违反度

    def penalty(self, g1):
        z = 0
        # 均为不等式约束，可以直接是否大于0直接求约束违反度
        for i in range(len(g1)):
            if g1[i] > 0:
                z += (g1[i])
        return z

    def punish(self, x1):

        # 得到约束条件
        g1 = self.constraints(x1)
        # 得到约束违反度
        p = self.penalty(g1)
        # 返回评估值
        return p


# G04
class G04:
    # The optimum solution is x* = (78, 33, 29.9952560256815985, 45, 36.7758129057882073)
    #   where f(x*) = -3.066553867178332e+004.

    def __init__(self):
        self.dimension = 5
        self.lower_bounds = [ 78, 33, 27, 27, 27]
        self.upper_bounds = [102, 45, 45, 45, 45]
        self.g = np.empty(6)
        # 保存惩罚因子大小
        self.alpha = 5000
        #     保存当前代数
        self.generation = 0


    def fitness(self, x):

        return 5.3578547 * x[2]**2 + 0.8356891 * x[0] * x[4] + 37.293239 * x[0] - 40792.141

    def constraints(self, x):
        u = 85.334407 + 0.0056858 * x[1] * x[4] + 0.0006262 * x[0] * x[3] - 0.0022053 * x[2] * x[4]
        v = 80.51249 + 0.0071317 * x[1] * x[4] + 0.0029955 * x[0] * x[1] + 0.0021813 * x[2] ** 2
        w = 9.300961 + 0.0047026 * x[2] * x[4] + 0.0012547 * x[0] * x[2] + 0.0019085 * x[2] * x[3]

        self.g[0] = -u
        self.g[1] = u - 92
        self.g[2] = -v + 90
        self.g[3] = v - 110
        self.g[4] = -w + 20
        self.g[5] = w - 25

        return self.g

    def evalution(self, x1):

        # 得到约束条件
        g1 = self.constraints(x1)
        # 得到约束违反度
        p = self.penalty(g1)
        # 得到目标函数值
        fit_ = self.fitness(x1)
        # 合并求和
        eval = fit_ + (self.alpha*(self.generation+1) * p)
        # 返回评估值
        return eval

        # 求解约束违反度

    def penalty(self, g1):
        z = 0
        # 均为不等式约束，可以直接是否大于0直接求约束违反度
        for i in range(len(g1)):
            if g1[i] > 0:
                z += (g1[i])
        return z

    def punish(self, x1):

        # 得到约束条件
        g1 = self.constraints(x1)
        # 得到约束违反度
        p = self.penalty(g1)
        # 返回评估值
        return p


# G05
class G05:
    # The optimum solution is x* = (679.945148297028709, 1026.06697600004691,
    #                                 0.118876369094410433, -0.39623348521517826)
    # where f(x*) = 5126.4967140071.

    def __init__(self):
        self.dimension = 4
        self.lower_bounds = [0, 0, -0.55, -0.55]
        self.upper_bounds = [1200, 1200, 0.55, 0.55]
        self.g = np.empty(5)
        # 记录约束变化
        self.deltaIni = self.dimension * np.log10(max(np.array(self.upper_bounds) - np.array(self.lower_bounds)))
        # 保存惩罚因子大小
        self.alpha = 5000
        #     保存当前代数
        self.generation = 0

    def fitness(self, x):

        return 3 * x[0] + 1e-6 * x[0]**3 + 2 * x[1] + 2e-6 / 3 * x[1]**3

    def constraints(self, x):

        self.g[0] = x[2] - x[3] - 0.55
        self.g[1] = x[3] - x[2] - 0.55
        self.g[2] = abs(1000 * np.sin(-x[2] - 0.25) + 1000 * np.sin(-x[3] - 0.25) + 894.8 - x[0]) -max(self.deltaIni/(1.015**self.generation),1e-4)
        self.g[3] =abs( 1000 * np.sin(x[2] - 0.25) + 1000 * np.sin(x[2] - x[3] - 0.25) + 894.8 - x[1]) -max(self.deltaIni/(1.015**self.generation),1e-4)
        self.g[4] = abs(1000 * np.sin(x[3] - 0.25) + 1000 * np.sin(x[3] - x[2] - 0.25) + 1294.8) -max(self.deltaIni/(1.015**self.generation),1e-4)

        return self.g


    def evalution(self, x1):

        # 得到约束条件
        g1 = self.constraints(x1)
        # 得到约束违反度
        p = self.penalty(g1)
        # 得到目标函数值
        fit_ = self.fitness(x1)
        # 合并求和
        eval = fit_ + (self.alpha*(self.generation+1) * p)
        # 返回评估值
        return eval

        # 求解约束违反度

    def penalty(self, g1):
        z = 0
        # 均为不等式约束，可以直接是否大于0直接求约束违反度
        for i in range(len(g1)):
            if g1[i] > 0:
                z += (g1[i])
        return z

    def punish(self, x1):

        # 得到约束条件
        g1 = self.constraints(x1)
        # 得到约束违反度
        p = self.penalty(g1)
        # 返回评估值
        return p
