# coding:UTF-8
# @Time: 2023/4/17 9:15
# @Author: Lulu Cao
# @File: advection.py
# @Software: PyCharm

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# 定义平流方程
def advection(t, u, c):
    du = np.zeros_like(u)
    du[1:-1] = -c * (u[2:] - u[:-2]) / (2 * dx)
    return du

# 设置参数
c = 1.0  # 平流速度
L = 1.0  # 域的长度
nx = 100  # 网格点数
dx = L / (nx - 1)  # 网格间距
x = np.linspace(0, L, nx)  # 网格点

# 设置初始条件
u0 = np.exp(-200 * (x - 0.5)**2)

# 求解平流方程
sol = solve_ivp(advection, (0, 1), u0, args=(c,), dense_output=True)

# 绘制解决方案
t = np.linspace(0, 1, 10)
for ti in t:
    plt.plot(x, sol.sol(ti))
plt.show()