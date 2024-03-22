# coding:UTF-8
# @Time: 2023/5/29 11:05
# @Author: Lulu Cao
# @File: PoissonDataset.py
# @Software: PyCharm
import random
import numpy as np
import math
def dataset1D(sample=10,x_min=0,x_max=1):
    # 准备数据
    # 生成一个符合pde关系的数据集,如果有解析解，生成符合解析解的也可以
    X_train = [random.uniform(x_min, x_max) for _ in range(sample)]
    X_train.append(0)
    X_train.append(1)
    X_train = np.array(X_train)

    y = []
    for x in X_train:
        y_temp = math.sin(math.pi * x)
        y.append(y_temp)
    return [X_train],y

def dataset2D(sample=10,x_min=0,x_max=1):
    # 创建空列表存储x1、x2和y的值
    x1_list = []
    x2_list = []
    y_list = []

    # 循环n次，随机生成n个采样点
    for i in range(sample):
        # 随机生成x1和x2的值
        x1 = random.uniform(x_min, x_max)
        x2 = random.uniform(x_min, x_max)
        # 计算y的值
        y = math.sin(math.pi * x1) * math.sin(math.pi * x2)
        # 将x1、x2和y的值添加到列表中
        x1_list.append(x1)
        x2_list.append(x2)
        y_list.append(y)


    return [x1_list,x2_list],y_list



def dataset3D(sample=10,x_min=0,x_max=1):
    # 创建空列表存储x1、x2和y的值
    x1_list = []
    x2_list = []
    x3_list = []
    y_list = []

    # 循环n次，随机生成n个采样点
    for i in range(sample):
        # 随机生成x1和x2的值
        x1 = random.uniform(x_min, x_max)
        x2 = random.uniform(x_min, x_max)
        x3 = random.uniform(x_min, x_max)
        # 计算y的值
        y = math.sin(math.pi * x1) * math.sin(math.pi * x2)* math.sin(math.pi * x3)
        # 将x1、x2和y的值添加到列表中
        x1_list.append(x1)
        x2_list.append(x2)
        x3_list.append(x3)
        y_list.append(y)


    return [x1_list,x2_list,x3_list],y_list


if __name__ == '__main__':
    x_min = 0
    x_max = 1
    # X1_train,X2_train,y = dataset2D(sample=10,x_min=x_min,x_max=x_max)
    X_train, y = dataset3D(sample=10, x_min=x_min, x_max=x_max)
    print(X_train[0][0])
    print(len(X_train[0]))
    print(X_train)
    print(y)