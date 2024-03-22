# coding:UTF-8
# @Time: 2023/4/23 10:10
# @Author: Lulu Cao
# @File: evaluate.py
# @Software: PyCharm
import math
from sympy import symbols, diff, lambdify,Derivative,Function,simplify,expand
import warnings
import numpy as np



# 定义一个适应度评估函数，使用均方误差作为评估指标，并接受两个参数，即个体和数据集
def evalSymbReg(individual,c,l,toolbox,X_train,y):
    expr = toolbox.compile(expr=individual)
    try:
        y_pred = [expr(X_train[i]) for i in range(len(X_train))]
        error = [(y_pred[i] - y[i]) ** 2 for i in range(len(X_train))]
    except ValueError:
        print(individual)
    mse = math.fsum(error) / len(error)
    # length = len(individual)


    rng = np.random.RandomState(0)
    X_train = rng.uniform(0, l, 100)
    pde_error1 = Euler_Bernoulli1(expr,c,X_train)

    pde_error2 = Euler_Bernoulli2(expr,l)
    return [pde_error1,pde_error2,mse]
    # return [pde_error1+pde_error2+len(individual)/10]
    # return [mse, length, pde_error]


def Euler_Bernoulli1(f,c,X_train):
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            # tensor = torch.from_numpy(arr)
            x = symbols('x')
            w = Function('w')(x)
            # 定义 Euler-Bernoulli 偏微分方程
            pde1 = Derivative(w, x, 4)-c
            # 将 lambda 函数转换为 SymPy 表达式
            w_expr = f(x)

            # 计算偏微分方程的值
            pde_value = pde1.subs(w, w_expr).doit()
            result = expand(pde_value)
            result = lambdify(x, result)
            error1 = [abs(result(X_train[i]))  for i in range(len(X_train))]

            return math.fsum(error1)
        except  Warning:
            return 1000


def Euler_Bernoulli2(f,l):
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            # tensor = torch.from_numpy(arr)
            x = symbols('x')
            w = Function('w')(x)
            # 定义 Euler-Bernoulli 偏微分方程
            pde2 = Derivative(w, x, 2)

            # 将 lambda 函数转换为 SymPy 表达式
            w_expr = f(x)

            # 计算偏微分方程的值
            pde_value = pde2.subs(w, w_expr).doit()
            result = simplify(pde_value)
            result = lambdify(x, result)
            error2 = abs(result(0))+abs(result(l)) #l
            return error2
        except  Warning:
            return 1000
        except ZeroDivisionError:
            return 1000

