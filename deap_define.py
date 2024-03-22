# coding:UTF-8
# @Time: 2023/4/20 15:38
# @Author: Lulu Cao
# @File: deap_define.py
# @Software: PyCharm
import operator
import math
import random

import numpy

from deap import algorithms # 提供了各种进化算法
from deap import base # 用于定义进化算法的基础类，如Fitness和Toolbox
from deap import creator # 用于动态创建新类，例如个体和种群
from deap import tools # 包含用于进化算法的运算符，例如选择、交叉和变异
from deap import gp

from scipy.optimize import minimize


# 用于在除数为零时返回1，以避免除零错误
def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1





def define_gp():
    """
    定义了用于生成表达式、个体和种群的函数，用于将树表达式转换为可调用函数的编译函数 等等
    :return: 进化过程中会用到的一些工具箱
    """
    # 创建了一个PrimitiveSet，并添加了一些基本运算符和函数，如加、减、乘、除和三角函数。
    # 它还添加了一个随机常数，并将输入参数重命名为’x’。



    pset = gp.PrimitiveSet("MAIN", 1) # 创建了一个名为 "MAIN" 的原语集，其中包含一个变量，名称默认为 "ARG0"
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(protectedDiv, 2)
    pset.addPrimitive(operator.neg, 1)
    pset.addPrimitive(math.cos, 1)
    pset.addPrimitive(math.sin, 1)
    # 临时常数是一种特殊类型的常数，它的值在每次创建新个体时都会重新生成
    # 添加了一个名称为 "RAND" 的临时常数。它的值由一个匿名函数生成，该函数使用 random.uniform(-2, 2) 来生成 [-2, 2] 范围内的随机数
    pset.addEphemeralConstant("RAND", lambda: random.uniform(-2, 2))
    pset.renameArguments(ARG0='x')





    creator.create("FitnessMin", base.Fitness, weights=(-1.0,)) # 创建了一个名为FitnessMin的新适应度类，用于最小化适应度值
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin) # 个体类，该类继承自gp.PrimitiveTree并具有新创建的适应度属性

    # 使用base.Toolbox创建了一个工具箱，并注册了一些用于生成表达式、个体和种群的函数。
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    # 它还注册了一个编译函数，用于将树表达式转换为可调用函数
    toolbox.register("compile", gp.compile, pset=pset)

    def evalSymbReg(individual, points):
        # Transform the tree expression into a callable function
        func = toolbox.compile(expr=individual)

        # Extract the constants from the individual
        constants = [node.value for node in individual if isinstance(node, gp.RAND)]

        # Define a function to compute the sum of squared errors
        def sum_of_squared_errors(constants):
            errors = [(func(x) - x**4 - x**3 - x**2 - x)**2 for x in points]
            return math.fsum(errors) / len(points),

        # Optimize the constants using another optimization algorithm
        optimized_constants = minimize(sum_of_squared_errors, constants)

        # Update the individual with the optimized constants
        k = 0
        for i, node in enumerate(individual):
            if isinstance(node, gp.Terminal) and node.name.startswith("ARG"):
                individual[i] = gp.Terminal(optimized_constants[k], False, node.ret)
                k += 1

        # Compute the sum of squared errors using the updated individual
        func = toolbox.compile(expr=individual)
        fitness = sum((y - func(x)) ** 2 for x, y in points)

        return fitness


    toolbox.register("evaluate", evalSymbReg, points=[x / 10. for x in range(-10, 10)]) # 将此评估函数注册到工具箱中

    # 注册了选择、交叉和变异运算符，并使用静态限制装饰器限制交叉和变异后树的高度。
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

    return toolbox








def main():
    random.seed(318)
    toolbox = define_gp()
    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(1) # 该对象用于存储种群中的最优个体。参数1表示它可以存储的最优个体数量。

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values) # 创建了一个统计对象，用于收集种群中个体适应度值的统计信息
    stats_size = tools.Statistics(len) # 用于收集种群中个体大小的统计信息
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size) # 创建了一个多重统计对象，用于同时收集种群中个体适应度值和大小的统计信息
    # 代码使用mstats.register方法注册了要计算的统计量：平均值、标准差、最小值和最大值统计量
    # 在遗传算法的每一代结束时，这个多重统计对象会自动更新以包含当前种群中个体的统计信息。
    # 可以使用logbook.record(**mstats.compile(pop))来记录当前种群中个体适应度值和大小的统计信息。
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)

    # 使用algorithms.eaSimple运行简单遗传算法，并返回结果。
    # 这个算法使用工具箱中注册的选择、交叉和变异运算符，并在每一代结束时更新统计信息和最优值。
    pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, 40, stats=mstats,
                                   halloffame=hof, verbose=True)
    # print log
    return pop, log, hof


if __name__ == "__main__":
    main()