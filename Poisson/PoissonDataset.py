# coding:UTF-8
# @Time: 2023/5/29 11:05
# @Author: Lulu Cao
# @File: PoissonDataset.py
# @Software: PyCharm
import random
import numpy as np
import math
def dataset1_1D(sample=10,x_min=0,x_max=1):
    # 准备数据
    random.seed(42)  # Set a seed for reproducibility
    X_train = [random.uniform(x_min, x_max) for _ in range(sample)]
    X_train = np.array(X_train)
    y = np.sin(np.pi*X_train)
    return [X_train], y.tolist()

def dataset1_2D(sample=10,x_min=0,x_max=1):
    # 准备数据
    random.seed(42)  # Set a seed for reproducibility
    x1 = [random.uniform(x_min, x_max) for _ in range(sample)]
    x1 = np.array(x1)
    random.seed(43)
    x2 = [random.uniform(x_min, x_max) for _ in range(sample)]
    x2 = np.array(x2)
    y = np.sin(np.pi*x1)*np.sin(np.pi*x2)
    return [x1,x2], y.tolist()

def dataset1_3D(sample=10,x_min=0,x_max=1):
    # 准备数据
    random.seed(42)  # Set a seed for reproducibility
    x1 = [random.uniform(x_min, x_max) for _ in range(sample)]
    x1 = np.array(x1)
    random.seed(43)
    x2 = [random.uniform(x_min, x_max) for _ in range(sample)]
    x2 = np.array(x2)
    random.seed(44)
    x3 = [random.uniform(x_min, x_max) for _ in range(sample)]
    x3 = np.array(x3)
    y = np.sin(np.pi*x1)*np.sin(np.pi*x2)*np.sin(np.pi*x3)
    return [x1,x2,x3], y.tolist()

def dataset2_1D(sample=10,x_min=0,x_max=1):
    # 准备数据
    random.seed(42)  # Set a seed for reproducibility
    x1 = [random.uniform(x_min, x_max) for _ in range(sample)]
    x1 = np.array(x1)
    y = 1/2-1/2*x1**2
    return [x1], y.tolist()

def dataset2_2D(sample=10,x_min=0,x_max=1):
    # 准备数据
    random.seed(42)  # Set a seed for reproducibility
    x1 = [random.uniform(x_min, x_max) for _ in range(sample)]
    x1 = np.array(x1)
    random.seed(43)
    x2 = [random.uniform(x_min, x_max) for _ in range(sample)]
    x2 = np.array(x2)
    y = 1/4-1/4*x1**2-1/4*x2**2
    return [x1,x2], y.tolist()

def dataset2_3D(sample=10,x_min=0,x_max=1):
    # 准备数据
    random.seed(42)  # Set a seed for reproducibility
    x1 = [random.uniform(x_min, x_max) for _ in range(sample)]
    x1 = np.array(x1)
    random.seed(43)
    x2 = [random.uniform(x_min, x_max) for _ in range(sample)]
    x2 = np.array(x2)
    random.seed(44)
    x3 = [random.uniform(x_min, x_max) for _ in range(sample)]
    x3 = np.array(x3)
    y = 1/6-1/6*x1**2-1/6*x2**2-1/6*x3**2
    return [x1,x2,x3], y.tolist()



if __name__ == '__main__':
    x_min = 0
    x_max = 1
    X_train, y = dataset2_1D(sample=10, x_min=x_min, x_max=x_max)
    print(X_train,y)