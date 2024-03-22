from scipy.optimize import minimize
from deap import gp
import dis
import inspect
import random

# 定义一个损失函数
def loss(params,ind,X1,y,pset):
    i = 0
    # print(params)
    for node in ind:
        # 检查节点是否为常数
        #if isinstance(node, gp.RAND) or (isinstance(node, gp.Terminal) and not isinstance(node.value, str)):
        if (isinstance(node, gp.Terminal) and not isinstance(node.value, str)):
            # 修改常数值
            node.value = params[i]
            i+=1
    ind.expr = gp.compile(ind, pset)
    # print(inspect.signature(ind.expr))

    y_pred = [ind.expr(X1[i]) for i in range(len(X1))]
    return sum([(a - b)**2 for a, b in zip(y, y_pred)])/len(y)

def local_optimize(individual, X, y,pset):
    """
    定义一个局部优化函数，它可以用scipy.optimize.minimize来调整个体中的常数系数
    :param individual:
    :param X:
    :param y:
    :param pset:
    :return:
    """

    params = [1 for node in individual if (isinstance(node, gp.Terminal) and not isinstance(node.value, str))]
    # node.value
    if len(params) == 0:
        return



    # 使用scipy.optimize.minimize来最小化损失函数

    bounds = [(-2, 2) for i in range(len(params))]
    minimize(loss, params,args=(individual, X, y,pset), method='SLSQP',bounds=tuple(bounds))


