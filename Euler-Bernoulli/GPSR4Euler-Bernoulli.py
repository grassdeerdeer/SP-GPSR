# coding:UTF-8
# @Time: 2023/4/28 8:35
# @Author: Lulu Cao
# @File: GPSR4Euler-Bernoulli.py
# @Software: PyCharm

# 导入deap和其他需要的库
from deap import base, creator, gp, tools, algorithms
import operator
import math
import random
import numpy as np
from local_optimize import *
from evaluate import *
from sympy import symbols,expand
import sympy
from parse_string import *
from sympy.parsing.sympy_parser import parse_expr




def dataset(c,l,sample=10):
    # 准备数据
    # 生成一个符合pde关系的数据集,如果有解析解，生成符合解析解的也可以
    X_train = [random.uniform(0, l) for _ in range(sample)]
    X_train.append(0)
    X_train.append(l)
    X_train = np.array(X_train)
    y = list(c/24*(X_train**4 - 2*l*X_train**3 + l**3*X_train))
    return X_train,y

def define_gp(X_train,y,name_list,pde_const):


    # 定义一个保护除法函数，避免除零错误
    def protectedDiv(left, right):
        try:
            return left / right
        except ZeroDivisionError:
            return 1


    pset = gp.PrimitiveSet("MAIN", 1) # 创建了一个名为 "MAIN" 的原语集，其中包含1个变量，名称默认为 "ARG0"
    pset.renameArguments(ARG0='x')

    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)

    pset.addTerminal(terminal=1.0)



    # 定义一个适应度类，继承自base.Fitness，并指定weights属性为负数，表示越小越好
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,-1.0,-1.0))

    # 定义一个个体类，继承自gp.PrimitiveTree，并添加fitness属性和params属性，并添加local_optimize方法
    creator.create("Individual", gp.PrimitiveTree,
                   fitness=creator.FitnessMin,
                   params=None,
                   local_optimize=local_optimize)


    # 创建一个基类对象，并注册相关的属性和方法
    toolbox = base.Toolbox()

    # 注册表达式生成器，使用ramped half-and-half方法，并指定最大深度为4
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=2, max_= 5 )

    # 注册个体生成器，使用表达式生成器，并将结果转换为个体类
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)

    # 注册种群生成器，使用个体生成器，并指定种群大小为1000
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # 它还注册了一个编译函数，用于将树表达式转换为可调用函数
    toolbox.register("compile", gp.compile, pset=pset)


    # 注册适应度评估函数，使用均方误差作为评估指标，并传入数据集作为参数
    toolbox.register("evaluate", evalSymbReg,toolbox=toolbox,X_train=X_train,y=y)

    # 注册选择算子
    toolbox.register("select", tools.selNSGA2)  # 使用NSGA-II选择算子
    toolbox.register("selbest", tools.selBest) # 如果你的适应度函数返回一个元组，表示多个目标值，那么tools.selBest会根据元组中的第一个值进行排序，然后选择最优的个体。


    # 注册交叉算子
    toolbox.register("leafmate", gp.cxOnePointLeafBiased, termpb=0.1)  # 叶子偏置的单点交叉算子，可以根据一个概率参数termpb来选择交换两棵树的内部节点或叶子节点。

    # 随机选择两棵树的一个内部节点，并交换它们的子树。使用静态限制装饰器，对交叉或变异算子添加一个限制条件，例如树的高度
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

    # 注册变异算子，然后对树中的每个节点，以一定的概率用 expr 生成的新表达式替换它。
    toolbox.register("expr_mut", gp.genFull, min_=1, max_=2)  # 使用完全生成法作为变异算子的基础
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)  # 使用均匀变异算子


    def from_string(expr_str, pset):
        tree = gp.PrimitiveTree.from_string(expr_str, pset)
        return tools.initIterate(creator.Individual, lambda: tree)

    toolbox.register("individual_from_string", from_string, pset=pset)
    return toolbox,pset



def init_gp(toolbox,pset,X_train,y,c,l):




    # 定义进化的参数，比如种群大小、进化代数、交叉概率、变异概率等
    popnum = 100
    ngen = 20
    cxpb = 0.6
    mutpb = 0.6
    vacb = 0.8


    # 创建一个种群，并初始化每个个体的参数和适应度值
    pop = toolbox.population(n=popnum)

    best_inds = []  # 记录每一代的最优个体
    hof = tools.HallOfFame(1)

    # 进行进化，每一代都进行选择、交叉、变异和评估，并记录统计信息
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)  # 创建一个多重统计对象，用来同时收集多个统计对象的数据
    mstats.register("avg", np.mean)  # 计算每个统计对象的平均值
    mstats.register("std", np.std)  # 标准差
    mstats.register("min", np.min)  # 最小值
    mstats.register("max", np.max)  # 最大值

    # 创建日志记录器
    logbook = tools.Logbook()

    for g in range(ngen):

        new_pop = toolbox.population(n=popnum)
        for i,ind in enumerate(best_inds):
           expr = gp.compile(ind, pset)
           x = symbols('x')
           expr = expr(x)
           simplified_expr = expand(expr)
           simplified_expr = str(simplified_expr)
           simplified_expr = convert_power(simplified_expr)
           simplified_expr = parse_expr(simplified_expr, evaluate=False)
           prefix_str = to_prefix(simplified_expr)
           prefix_str = prefix_str.replace("Add", "add")
           prefix_str = prefix_str.replace("Sub", "sub")
           prefix_str = prefix_str.replace("Mul", "mul")
           ind = toolbox.individual_from_string(prefix_str)
           ind.local_optimize(X_train, y, pset)
           # 评估个体的适应度值
           ind.fitness.values = toolbox.evaluate(ind, c, l)
           best_inds[i] = ind

        pop[:] = pop + new_pop + best_inds


        for i, ind in enumerate(pop):
            ind.local_optimize(X_train, y, pset)
            # 评估个体的适应度值
            ind.fitness.values = toolbox.evaluate(ind,c,l)
            pop[i] = ind


        # 选择下一代的个体
        # offspring = toolbox.selbest(pop, popnum)
        offspring = toolbox.select(pop, popnum)
        # 克隆每个个体，避免修改原始种群
        offspring = list(map(toolbox.clone, offspring))

        # 对选出的个体进行交叉和变异
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < mutpb:
                if len(mutant)<=1:
                    continue
                toolbox.mutate(mutant)
                del mutant.fitness.values

        for index,vacind in enumerate(offspring):
            if random.random()<vacb:
                expr = gp.compile(vacind, pset)
                x = symbols('x')
                expr = expr(x)
                simplified_expr = expand(expr)
                simplified_expr = str(simplified_expr)
                simplified_expr = convert_power(simplified_expr)
                simplified_expr = parse_expr(simplified_expr, evaluate=False)
                prefix_str = to_prefix(simplified_expr)
                prefix_str = prefix_str.replace("Add", "add")
                prefix_str = prefix_str.replace("Sub", "sub")
                prefix_str = prefix_str.replace("Mul", "mul")
                ind = toolbox.individual_from_string(prefix_str)


                params = [node.value for node in ind if
                          (isinstance(node, gp.Terminal) and not isinstance(node.value, str))]
                if len(params) == 0:
                    continue
                index = random.randint(0, len(params) - 1)  # 随机生成一个列表索引
                params[index] = 0  # 将该索引对应的元素设为0
                i = 0
                for node in ind:
                    # 检查节点是否为常数
                    # if isinstance(node, gp.RAND) or (isinstance(node, gp.Terminal) and not isinstance(node.value, str)):
                    if (isinstance(node, gp.Terminal) and not isinstance(node.value, str)):
                        # 修改常数值
                        node.value = params[i]
                        i += 1

                expr = gp.compile(ind, pset)
                x = symbols('x')
                expr = expr(x)
                simplified_expr = expand(expr)
                simplified_expr = str(simplified_expr)
                simplified_expr = convert_power(simplified_expr)
                simplified_expr = parse_expr(simplified_expr, evaluate=False)
                prefix_str = to_prefix(simplified_expr)
                prefix_str = prefix_str.replace("Add", "add")
                prefix_str = prefix_str.replace("Sub", "sub")
                prefix_str = prefix_str.replace("Mul", "mul")
                ind = toolbox.individual_from_string(prefix_str)
                del ind.fitness.values
                offspring[index] = ind



        # 对每个新生成的个体进行局部优化和评估
        for i,ind in enumerate(offspring):
            if not ind.fitness.valid:
                # 编译个体的表达式为一个可执行的函数，并赋值给expr属性
                ind.local_optimize(X_train, y, pset)
                # 评估个体的适应度值
                ind.fitness.values = toolbox.evaluate(ind, c, l)
                offspring[i]=ind

        # 更新种群
        pop[:] = pop + offspring
        fitness_set = set()  # 创建一个空集合
        new_pop = []  # 创建一个新的种群列表
        for ind in pop:  # 遍历种群中的每个个体
            if ind.fitness not in fitness_set:  # 如果个体的适应度值不在集合中
                fitness_set.add(ind.fitness)  # 将适应度值添加到集合中
                new_pop.append(ind)  # 将个体添加到新的种群列表中
        pop = new_pop  # 用新的种群列表替换原来的种群列表
        # offspring = toolbox.selBest(pop, popnum)
        offspring = toolbox.select(pop, popnum)
        pop[:] = offspring


        # 保存最优个体
        hof.update(pop)
        best_individual = hof.items[0]
        best_inds.append(best_individual)
        best_ind = min(pop, key=lambda ind: ind.fitness.values[0])
        expr = gp.compile(best_ind, pset)
        x = symbols('x')
        expr = expr(x)
        simplified_expr = expand(expr)

        # 记录当前种群的统计信息
        record = mstats.compile(pop)
        logbook.record(gen=g, **record)
        print(f"Generation {g}: {str(simplified_expr)}")
        # 打印日志记录器
    print(logbook)
    return pop

def evolution_gp(toolbox,pset,X_train,y, c, l,pop):
    # 定义进化的参数，比如种群大小、进化代数、交叉概率、变异概率等
    popnum = 100
    ngen = 1
    cxpb = 0.8
    mutpb = 0.6
    vacb = 0.6

    # 已有初始种群

    best_inds = []  # 记录每一代的最优个体
    hof = tools.HallOfFame(1)







    # 进行进化，每一代都进行选择、交叉、变异和评估，并记录统计信息
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size) # 创建一个多重统计对象，用来同时收集多个统计对象的数据
    mstats.register("avg", np.mean)
    mstats.register("std", np.std) # 标准差
    mstats.register("min", np.min) # 最小值
    mstats.register("max", np.max) # 最大值

    # 创建日志记录器
    logbook = tools.Logbook()


    for g in range(ngen):

        new_pop = toolbox.population(n=popnum)
        # for i,ind in enumerate(best_inds):
        #     expr = gp.compile(ind, pset)
        #     x = symbols('x')
        #     expr = expr(x)
        #     simplified_expr = expand(expr)
        #     simplified_expr = str(simplified_expr)
        #     simplified_expr = convert_power(simplified_expr)
        #     simplified_expr = parse_expr(simplified_expr, evaluate=False)
        #     prefix_str = to_prefix(simplified_expr)
        #     prefix_str = prefix_str.replace("Add", "add")
        #     prefix_str = prefix_str.replace("Sub", "sub")
        #     prefix_str = prefix_str.replace("Mul", "mul")
        #     ind = toolbox.individual_from_string(prefix_str)
        #     ind.local_optimize(X_train, y, pset)
        #     # 评估个体的适应度值
        #     ind.fitness.values = toolbox.evaluate(ind, c, l)
        #     best_inds[i] = ind

        pop[:] = pop + new_pop + best_inds


        for i, ind in enumerate(pop):
            ind.local_optimize(X_train, y, pset)
            # 评估个体的适应度值
            ind.fitness.values = toolbox.evaluate(ind,c,l)
            pop[i] = ind


        # 选择下一代的个体
        # offspring = toolbox.selbest(pop, popnum)
        offspring = toolbox.select(pop, popnum)
        # 克隆每个个体，避免修改原始种群
        offspring = list(map(toolbox.clone, offspring))

        # 对选出的个体进行交叉和变异
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < mutpb:
                if len(mutant)<=1:
                    continue
                toolbox.mutate(mutant)
                del mutant.fitness.values

        for index,vacind in enumerate(offspring):
            if random.random()<vacb:
                expr = gp.compile(vacind, pset)
                x = symbols('x')
                expr = expr(x)
                simplified_expr = expand(expr)
                simplified_expr = str(simplified_expr)
                simplified_expr = convert_power(simplified_expr)
                simplified_expr = parse_expr(simplified_expr, evaluate=False)
                prefix_str = to_prefix(simplified_expr)
                prefix_str = prefix_str.replace("Add", "add")
                prefix_str = prefix_str.replace("Sub", "sub")
                prefix_str = prefix_str.replace("Mul", "mul")
                ind = toolbox.individual_from_string(prefix_str)


                params = [node.value for node in ind if
                          (isinstance(node, gp.Terminal) and not isinstance(node.value, str))]
                if len(params) == 0:
                    continue
                index = random.randint(0, len(params) - 1)  # 随机生成一个列表索引
                params[index] = 0  # 将该索引对应的元素设为0
                i = 0
                for node in ind:
                    # 检查节点是否为常数
                    # if isinstance(node, gp.RAND) or (isinstance(node, gp.Terminal) and not isinstance(node.value, str)):
                    if (isinstance(node, gp.Terminal) and not isinstance(node.value, str)):
                        # 修改常数值
                        node.value = params[i]
                        i += 1

                expr = gp.compile(ind, pset)
                x = symbols('x')
                expr = expr(x)
                simplified_expr = expand(expr)
                simplified_expr = str(simplified_expr)
                simplified_expr = convert_power(simplified_expr)
                simplified_expr = parse_expr(simplified_expr, evaluate=False)
                prefix_str = to_prefix(simplified_expr)
                prefix_str = prefix_str.replace("Add", "add")
                prefix_str = prefix_str.replace("Sub", "sub")
                prefix_str = prefix_str.replace("Mul", "mul")
                ind = toolbox.individual_from_string(prefix_str)
                del ind.fitness.values
                offspring[index] = ind



        # 对每个新生成的个体进行局部优化和评估
        for i,ind in enumerate(offspring):
            if not ind.fitness.valid:
                # 编译个体的表达式为一个可执行的函数，并赋值给expr属性
                ind.local_optimize(X_train, y, pset)
                # 评估个体的适应度值
                ind.fitness.values = toolbox.evaluate(ind, c, l)
                offspring[i]=ind

        # 更新种群
        pop[:] = pop + offspring
        fitness_set = set()  # 创建一个空集合
        new_pop = []  # 创建一个新的种群列表
        for ind in pop:  # 遍历种群中的每个个体
            if ind.fitness not in fitness_set:  # 如果个体的适应度值不在集合中
                fitness_set.add(ind.fitness)  # 将适应度值添加到集合中
                new_pop.append(ind)  # 将个体添加到新的种群列表中
        pop = new_pop  # 用新的种群列表替换原来的种群列表
        # offspring = toolbox.selBest(pop, popnum)
        offspring = toolbox.select(pop, popnum)
        pop[:] = offspring


        # 保存最优个体
        hof.update(pop)
        best_individual = hof.items[0]
        best_inds.append(best_individual)
        best_ind = min(pop, key=lambda ind: ind.fitness.values[0])
        expr = gp.compile(best_ind, pset)
        x = symbols('x')
        expr = expr(x)
        simplified_expr = expand(expr)

        # 记录当前种群的统计信息
        record = mstats.compile(pop)
        logbook.record(gen=g, **record)
        print(f"Generation {g}: {str(simplified_expr)}")
    # 打印日志记录器
    print(logbook)

def main():



    time = 30
    
    # 1. 采样数据集
    c = 5e-0 #random.uniform(0, 1)
    l = 2 #random.randint(0, 1)
    sample =11 #+2
    X_train, y_train = dataset(c, l, sample)
    # 2. 定义变量
    name_list = ["x"]
    toolbox, pset = define_gp(X_train, y_train, name_list, c)
    for t in range(time):
        
        #if t==0:
            
        initpop = init_gp(toolbox,pset,X_train,y_train,c,l)
        #else:
            # 删除原来注册的 evaluate 函数
            #toolbox.unregister("evaluate")
            #toolbox.register("evaluate", evalSymbReg, X_train=X_train, y=y_train, toolbox=toolbox)
            #evolution_gp(toolbox, pset, X_train, y_train,c,l,initpop)





if __name__ == "__main__":
    main()
