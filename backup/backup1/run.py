import numpy as np
from standardPSO import PSO_Standard
from  env import env_PSO
from Policy_Gradient import PolicyGradient
from  matplotlib import pyplot as plt
from CEC2006 import G01,G02,G03,G04,G05

if __name__=="__main__":
    #1 初始化一些参数

    # 定义一个episode的迭代次数
    iteration_times = 500
    # 定义种群数目
    population_num=500
    # 定义初始参数
    w=0.9
    c1=2.05
    c2=2.05
    problem_01=G01()
    problem_02=G02()
    problem_03=G03()
    problem_04=G04()
    problem_05=G05()

    now_problem=problem_03


    #2 初始化模型和强化学习网络

    env=env_PSO(population_num,iteration_times,now_problem,c1,c2,w)
    RL = PolicyGradient(n_actions=env.action_space.n, n_features=3, learning_rate=0.2, reward_decay=0.95,
                        output_graph=False)

    #3 记录强化学习过程中，最好的收敛曲线
    RL_best_curve=None
    # 记录强化学习每个episode的最优函数值
    RL_best_fit=float('inf')
    # 记录最优位置
    RL_best_pos=None


    # 4 每一个episode进行迭代
    for i_episode in range(10):
        observation = env.reset(c1,c2,w)

        # 没有完成一个episode前，一直循环
        while True:
            action = RL.choose_action(observation)

            observation_, reward, done, info = env.step(action)

            RL.store_transition(observation, action, reward)

            if done:

                ep_rs_sum = sum(RL._discount_and_norm_rewards())

                running_reward = ep_rs_sum

                print(running_reward)

                print("episode:", i_episode, " , best_fitness", env.gbestFit)

                if env.gbestFit <= RL_best_fit:
                    RL_best_fit = env.gbestFit
                    RL_best_curve = env.fitness_list
                    RL_best_pos=env.gbest

                vt = RL.learn()
                break

            observation = observation_

    # RL PSO最优解
    print('最优解位置',RL_best_pos)
    print('约束违反情况',env.fitnessFunction.constraints(RL_best_pos))
    print('最优解大小',env.fitnessFunction.fitness(RL_best_pos))




    standard_best=float('inf')
    s_curve =None
    # 初始化标准PSO
    for i in range(10):
        s_pso = PSO_Standard(population_num,iteration_times,now_problem,c1,c2,w)
        s_pso.done()
        print('标准PSO，第',i,'次 ',s_pso.gbestFit)
        if s_pso.gbestFit<standard_best:
            standard_best=s_pso.gbestFit
            s_curve=s_pso.fitness_list


    # 绘制标准PSO收敛曲线
    iter_ = [i for i in range(iteration_times)]
    plt.plot(iter_, s_curve, label='standard_PSO', color='r')

    # 绘制强化学习PSO首先曲线
    plt.plot(iter_, RL_best_curve, label='RL_PSO', color='b')
    plt.legend()
    plt.show()