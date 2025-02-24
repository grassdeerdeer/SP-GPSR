# coding:UTF-8
# @Time: 2023/5/29 9:31
# @Author: Lulu Cao
# @File: evaluate.py
# @Software: PyCharm

# coding:UTF-8
# @Time: 2023/4/23 10:10
# @Author: Lulu Cao
# @File: evaluate.py
# @Software: PyCharm
import math
from sympy import symbols, diff, lambdify,Derivative,Function,simplify,expand
import warnings
import numpy as np
import sympy



# 定义一个适应度评估函数，使用均方误差作为评估指标，并接受两个参数，即个体和数据集
def evalSymbReg(individual,toolbox,X_train,y,d):
    expr = toolbox.compile(expr=individual)
    rng = np.random.RandomState(0)
    X_pde_train = rng.uniform(0, 1, 100)


    if d == 1:
        try:
            y_pred = [expr(X_train[0][i]) for i in range(len(X_train[0]))]
            error = [(y_pred[i] - y[i]) ** 2 for i in range(len(X_train[0]))]
        except ValueError:
            print(individual)
        mse = math.fsum(error) / len(error)

        pde_error1 = Poisson1d_1(expr,X_pde_train)
        pde_error2 = Poisson1d_2(expr, X_pde_train)

    elif d == 2:
        try:
            y_pred = [expr(X_train[0][i],X_train[1][i]) for i in range(len(X_train[0]))]
            error = [(y_pred[i] - y[i]) ** 2 for i in range(len(X_train[0]))]
        except ValueError:
            print(individual)
        mse = math.fsum(error) / len(error)
        pde_error1 = Poisson2d_1(expr, X_pde_train)
        pde_error2 = Poisson2d_2(expr, X_pde_train)

    elif d == 3:
        try:
            y_pred = [expr(X_train[0][i],X_train[1][i],X_train[2][i]) for i in range(len(X_train[0]))]
            error = [(y_pred[i] - y[i]) ** 2 for i in range(len(X_train[0]))]
        except ValueError:
            print(individual)
        mse = math.fsum(error) / len(error)
        pde_error1 = Poisson3d_1(expr, X_pde_train)
        pde_error2 = Poisson3d_2(expr, X_pde_train)




    #return [pde_error1,pde_error2,mse]
    # return [pde_error1+pde_error2+len(individual)/10]
    # return [mse, length, pde_error]
    return [pde_error1+pde_error2+mse]


def Poisson1d_1(f,X_train):
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            # tensor = torch.from_numpy(arr)
            x1 = symbols('x1')
            w = Function('w')(x1) # 定义一个函数
            # 定义 1DPoisson 偏微分方程
            pde1 = Derivative(w, x1, 2)+sympy.pi**2 * sympy.sin(sympy.pi * x1)
            # 将 lambda 函数转换为 SymPy 表达式
            w_expr = simplify(f(x1))

            # 计算偏微分方程的值
            pde_value = pde1.subs(w, w_expr).doit()
            result = expand(pde_value)
            result = lambdify(x1, result)
            error1 = [abs(result(X_train[i]))  for i in range(len(X_train))]

            return math.fsum(error1)
        except  Warning:
            return 1000


def Poisson1d_2(f,X_train):
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            return abs(f(0))+abs(f(1))
        except  Warning:
            return 1000
        except ZeroDivisionError:
            return 1000


def Poisson2d_1(f,X_train):
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            # 定义两个符号变量
            x1 = symbols('x1')
            x2 = sympy.Symbol('x2')

            # 定义 2DPoisson 偏微分方程
            w = Function('w')(x1,x2)

            pde1 = Derivative(w, x1, 2)+Derivative(w, x2, 2)+2*sympy.pi**2 * sympy.sin(sympy.pi * x1) * sympy.sin(sympy.pi * x2)
            # 将 lambda 函数转换为 SymPy 表达式
            w_expr = simplify(f(x1,x2))

            # 计算偏微分方程的值
            pde_value = pde1.subs(w, w_expr).doit()
            result = expand(pde_value)
            result = lambdify([x1,x2], result)
            error1 = [abs(result(X_train[i],X_train[i]))  for i in range(len(X_train))]

            return math.fsum(error1)
        except  Warning:
            return 1000


def Poisson2d_2(f,X_train):
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            x1b0 = [abs(f(0,X_train[i]))  for i in range(len(X_train))]
            x1b1 = [abs(f(1, X_train[i])) for i in range(len(X_train))]

            x2b0 = [abs(f(X_train[i],0)) for i in range(len(X_train))]
            x2b1 = [abs(f(X_train[i],1)) for i in range(len(X_train))]
            return math.fsum(x1b0)+math.fsum(x1b1)+math.fsum(x2b0)+math.fsum(x2b1)
        except  Warning:
            return 1000
        except ZeroDivisionError:
            return 1000



def Poisson3d_1(f,X_train):
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            # 定义两个符号变量
            x1 = symbols('x1')
            x2 = symbols('x2')
            x3 = symbols('x3')

            # 定义 2DPoisson 偏微分方程
            w = Function('w')(x1,x2,x3)

            pde1 = Derivative(w, x1, 2)+Derivative(w, x2, 2)+Derivative(w, x3, 2)+3*sympy.pi**2 * sympy.sin(sympy.pi * x1) * sympy.sin(sympy.pi * x2)* sympy.sin(sympy.pi * x3)
            # 将 lambda 函数转换为 SymPy 表达式
            w_expr = simplify(f(x1,x2,x3))

            # 计算偏微分方程的值
            pde_value = pde1.subs(w, w_expr).doit()
            result = expand(pde_value)
            result = lambdify([x1,x2,x3], result)
            error1 = [abs(result(X_train[i],X_train[i],X_train[i]))  for i in range(len(X_train))]

            return math.fsum(error1)
        except  Warning:
            return 1000


def Poisson3d_2(f,X_train):
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            x1b0 = [abs(f(0,X_train[i],X_train[i]))  for i in range(len(X_train))]
            x1b1 = [abs(f(1, X_train[i],X_train[i])) for i in range(len(X_train))]

            x2b0 = [abs(f(X_train[i],0,X_train[i])) for i in range(len(X_train))]
            x2b1 = [abs(f(X_train[i],1,X_train[i])) for i in range(len(X_train))]

            x3b0 = [abs(f(X_train[i],X_train[i], 0)) for i in range(len(X_train))]
            x3b1 = [abs(f(X_train[i],X_train[i], 1)) for i in range(len(X_train))]
            return math.fsum(x1b0)+math.fsum(x1b1)+math.fsum(x2b0)+math.fsum(x2b1)+math.fsum(x3b0)+math.fsum(x3b1)
        except  Warning:
            return 1000
        except ZeroDivisionError:
            return 1000
