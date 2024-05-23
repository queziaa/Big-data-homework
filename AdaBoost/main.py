from function import *
import numpy as np
from matplotlib import pyplot as plt

# 学习M个基本分类器，最终的分类器是这M个分类器的线性组合
def train_with_M(x,y,M=20):
    # 样本数量
    N = len(x)
    # 由M各基本分类器组成的列表。每个基本分类器对象包含其阈值的值/方向/所在维度，以及其在线性组合中的系数
    list_of_v = []
    # 初始化N个权重值分布
    w_ini = np.ones(N) * 1 / N
    wm = np.copy(w_ini)
    # Adaboost主循环
    for m in range(M):
        v,wm=Adaboost_Loop(x,y,N,wm)
        list_of_v.append(v)
    # 计算使用最终分类器的错误分类数
    total_error=G_errate(x,y,N,list_of_v)
    return list_of_v,total_error

# 遍历不同大小的M(M<=M_max),使得由M个分类器组成的最终的分类器能够使得错误分类数为0，求出满足这样条件的最小的M
def mintrain(x,y,M_max=20):
    # 样本数量
    N=len(x)
    M=0
    # 由M各基本分类器组成的列表。每个基本分类器对象包含其阈值的值/方向/所在维度，以及其在线性组合中的系数
    list_of_v=[]
    for M in range(1, M_max):
        # 测试每个M
        # 初始化N个权重值分布
        w_ini = np.ones(N) * 1 / N
        wm = np.copy(w_ini)
        # Adaboost主循环
        for m in range(M):
            v, wm = Adaboost_Loop(x, y, N, wm)
            list_of_v.append(v)
        # 计算此时得到的最终分类器在数据集上的错误分类数
        total_error=G_errate(x,y,N,list_of_v)
        # 如果错误分类数是0，则结束寻找M
        if total_error == 0:
            break
        else:list_of_v = []
    return list_of_v,M

# 训练与画图
def train_and_draw():
    # 加载鸢尾花数据集
    irisx,irisy=create_data()
    # irisx：N*2矩阵，N是样本点的数量，每个样本店是一个二维坐标里的点
    # irisy：either +1 or -1
    list_of_v,total_error=train_with_M(irisx,irisy)
    # list_of_v,M=mintrain(irisx,irisy)
    draw(irisx,irisy,list_of_v,'20.jpg')
    # draw(irisx,irisy,list_of_v,'min.jpg')

# 主函数
if __name__ == '__main__':
    train_and_draw()