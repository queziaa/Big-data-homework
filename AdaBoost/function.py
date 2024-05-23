import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

# 鸢尾花(iris)数据集
# 数据集内包含 3 类共 150 条记录，每类各 50 个数据，
# 每条记录都有 4 项特征：花萼长度、花萼宽度、花瓣长度、花瓣宽度，
# 可以通过这4个特征预测鸢尾花卉属于（iris-setosa, iris-versicolour, iris-virginica）中的哪一品种。
# 这里只取前100条记录，两项特征，两个类别。
def create_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    data = np.array(df.iloc[:100, [0, 1, -1]])
    for i in range(len(data)):
        if data[i,-1] == 0:
            data[i,-1] = -1
    return data[:,:2], data[:,-1]

# 辅助函数：抽取二维数据的第一维度和第二维度
def extract12(x):
    N=len(x)
    x1,x2=[],[]
    # 第一维度
    for i in range(N):
        x1.append(x[i][0])
    # 第二维度
    for i in range(N):
        x2.append(x[i][1])
    return x1,x2

# 用一个类对象存储二维样本弱分类器的阈值
# 下面会说明，二维样本弱分类器的阈值有维度、值本身、方向、对应的最小分类错误率四个属性
# 最终得到的分类器本身是弱分类器的线性组合，因此需要再保存一个线性系数alpha
class dv:
    def __init__(self,dim=0,val=0.0,dire=1,err=1,alpha=0):
        # 本类对象中的属性都是private的，故属性名用两个下划线开头
        # 弱分类器的阈值所属于的维度
        self.__dim=dim
        # 阈值本身
        self.__val=val
        # 阈值方向，如果是1则大于阈值属于正类，是-1则相反
        self.__dire=dire
        # 用于储存在求弱分类器时当前得到的最小错误分类率
        self.__err=err
        # 该弱分类器的线性组合系数
        self.__a=alpha
    # 设置阈值所属于的维度
    def setdim(self,dim):
        self.__dim=dim
    # 设置阈值本身
    def setval(self,val):
        self.__val=val
    # 设置阈值的方向
    def setdire(self,dire):
        self.__dire=dire
    # 更新（最小）错误分类率
    def seterr(self,err):
        self.__err=err
    # 设置弱分类器的线性组合系数
    def setalpha(self,a):
        self.__a=a
    # 获得阈值所属维度的接口函数
    def getdim(self):
        return self.__dim
    # 获得阈值本身
    def getval(self):
        return self.__val
    # 获得阈值的方向
    def getdire(self):
        return self.__dire
    # 获得阈值的（最小）错误率
    def geterr(self):
        return self.__err
    # 获得该弱分类器的线性组合系数
    def getalpha(self):
        return self.__a
    # 用来print
    def __str__(self):
        return 'Threshold='+str(self.__val)+',Direction='+str(self.__dire)+',Dimension='+str(self.__dim)

# 寻找基本分类器
# 一维的弱分类器的求法：只需要遍历一系列阈值，找到能够让错误分类率最小的阈值就可以
# 二维的弱分类器的求法比较复杂，先在第一个维度x1上寻找使得错误分类率最小的阈值
# 但此时找到的阈值带有三个特性：一是该阈值是在第一个维度上的阈值，二是该阈值的方向（是大于阈值为正类还是相反），三是此时的最小错误分类率
# 然后在第二个维度x2上寻找阈值，但如果在第二个维度上找到一个阈值使得此时的错误分类率小于之前得到的阈值，更新它
# 因此最终得到的阈值有四个参数：维度、值、方向、此时的最小分类错误率
# direction为1表示大于阈值是正类，为-1表示小于阈值是正类
def find_weak_classifier(x,y,D):
    # 初始化阈值为x1维度上的，值为0，方向1，最小分类错误率为1，线性组合系数为0，见类dv的定义
    DV=dv()
    # 抽取x1/x2维度
    x1,x2=extract12(x)
    # 先遍历x1,direction=1上的阈值
    for v in range(0,100):
        # 从0到10，间隔是0.1
        v = float(v / 10)
        # 测试当前维度、方向下的值为v的阈值的错误分类率
        err_now=count_errate(x1,y,D,v,dire=1)
        # 如果这个错误分类率不高于之前得到的最小错误分类率
        if err_now<=DV.geterr():
            # 更新最小错误分类率
            DV.seterr(err_now)
            # 更新阈值的值本身，此时还不用考虑是否要更新方向和维度
            DV.setval(v)
    # 遍历x1,direction=-1上的阈值
    for v in range(0,100):
        # 从0到10，间隔是0.1
        v=float(v/10)
        # 测试当前维度、方向下的值为v的阈值的错误分类率
        err_now=count_errate(x1,y,D,v,dire=-1)
        # 如果这个错误分类率不高于之前得到的最小错误分类率
        if err_now<=DV.geterr():
            # 更新最小错误分类率
            DV.seterr(err_now)
            # 更新阈值的值本身，此时还不用考虑是否要更新维度
            DV.setval(v)
            # 更新阈值的方向
            DV.setdire(-1)
    # 遍历x2,direction=1阈值
    for v in range(0,100):
        # v从0到10，间隔是0.1
        v=float(v/10)
        # 测试当前维度、方向下的值为v的阈值的错误分类率
        err_now=count_errate(x2,y,D,v,dire=1)
        # 如果这个错误分类率不高于之前得到的最小错误分类率
        if err_now<=DV.geterr():
            # 更新最小错误分类率
            DV.seterr(err_now)
            # 更新阈值的值本身
            DV.setval(v)
            # 更新阈值的方向
            DV.setdire(1)
            # 更新阈值的维度
            DV.setdim(1)
    # 遍历x2,direction=-1阈值
    for v in range(0, 100):
        # 从0到10，间隔是0.1
        v = float(v / 10)
        # 测试当前维度、方向下的值为v的阈值的错误分类率
        err_now = count_errate(x2, y, D, v, dire=-1)
        # 如果这个错误分类率不高于之前得到的最小错误分类率
        if err_now <= DV.geterr():
            # 更新最小错误分类率
            DV.seterr(err_now)
            # 更新阈值
            DV.setval(v)
            # 更新阈值方向
            DV.setdire(-1)
            # 更新阈值所在的维度
            DV.setdim(1)
    return DV

# 方向为dire（1为大于阈值判为正类，-1则相反），阈值为v的一维基本分类器Gm(x)
def Gm(x,v,dire):
    # 方向为1
    if dire==1:
        # 则大于阈值为正类
        if x>=v:
            return 1
        elif x<v:
            return -1
    # 方向为-1时，结果与方向为1时相反
    elif dire==-1:
        return (-1)*Gm(x,v,1)
    # 不是1或者-1时
    else:
        # 抛出异常
        raise ValueError('The direction should be either 1 or -1.')

# 计算权值为D的数据集x-y在阈值为v、方向为dire（+1表示大于阈值为正类）的一维基本分类器上得到的分类错误率。
def count_errate(x,y,D,v,dire):
    # 用于累计错误
    total_error=0
    # 样本数量
    N=len(x)
    for i in range(N):
        # 累计的错误率=（样本分类错误*该样本点的权值D[i]）的总和。
        # (Gm(x[i],v,dire)!=y[i])表示样本分类错误。
        total_error=total_error+(Gm(x[i],v,dire)!=y[i])*D[i]
    return total_error

# 对应统计学习方法157页的f(x),也就是基本分类器的线性组合
def f(x,list_of_v):
    y=0
    for v in list_of_v:
        # alpha（线性分类器的系数）*线性分类器的结果
        y=y+v.getalpha()*Gm(x[v.getdim()],v.getval(),v.getdire())
    return y

# 对f(x)取符号函数就是最终分类器G(x)
def G(x,list_of_v):
    return np.sign(f(x,list_of_v))

# 计算经过最终分类器（基本分类器的线性组合）的错误分类数
def G_errate(x,y,N,list_of_v):
    total_error = 0
    for i in range(N):
        total_error = total_error + (G(x[i], list_of_v) != y[i])
    return total_error

# Adaboost算法的循环部分（每一次循环找到一个基本分类器）
# x是二维样本，y是样本的所属类，N是样本的数量，wm是每次循环需要更新的样本权值
def Adaboost_Loop(x,y,N,wm):
    # 根据每个样本的权值D=wm，找到基本分类器之一
    v = find_weak_classifier(x, y, D=wm)
    # 此时的带权值分类错误率
    em = v.geterr()
    # 计算该基本分类器的线性组合系数am
    am = (1 / 2) * np.log((1 - em) / em)
    # 设置该基本分类器的系数
    v.setalpha(am)
    # 储存新权值
    w_new = np.zeros(N)
    # 以下计算新权值，参考统计学习方法157页
    e = np.zeros(N)
    for i in range(N):
        e[i] = np.exp((-1) * am * y[i] * Gm(x[i][v.getdim()], v.getval(), v.getdire()))
    Zm = 0
    for i in range(N):
        Zm = Zm + wm[i] * e[i]
    for i in range(N):
        w_new[i] = (wm[i] * e[i]) / Zm
    # 新权值
    wm = np.copy(w_new)
    # 打印基本分类器的特性：阈值，方向，维度
    print(v)
    # 返回本次循环得到的基本分类器特性以及新权值
    return v,wm

# 画样本点以及最终分类器的分割线
def draw(x,y,list_of_v,title):
    # 样本点颜色
    label = []
    # 将鸢尾花数据转换成列表
    xlist = x.tolist()
    ylist = y.tolist()
    # 由于样本点是二维的，提取其两个维度
    ix1, ix2 = extract12(xlist)
    # 设置样本点颜色，正类是红色，负类是蓝色
    for i in range(len(xlist)):
        if ylist[i] > 0:
            label.append('r')
        else:
            label.append('b')
    # 样本点散点
    plt.scatter(ix1, ix2, color=label)
    # 最终分类器分割线经过的点的坐标
    contour_list = []
    # 遍历包含所有样本点的最小矩形区域，根据“某一点被分类器判为正类，其邻居被判为负类”来判断某点是否在分割线上。
    # 由于鸢尾花数据集的正类整体在右侧，负类在左侧，因此认为右边一点是正类的负类点在边界上
    # 遍历的间隔是1/500,此时分割线是水平和垂直线
    prec=500
    x1_min, x1_max, x2_min, x2_max = int(prec * min(ix1)), int(prec * max(ix1)), int(prec * min(ix2)), int(prec * max(ix2))
    for j in range(x2_min, x2_max + 1):
        for i in range(x1_min, x1_max + 1):
            temp = [i / prec, j / prec]
            temp_next = [(i + 1) / prec, j / prec]
            if f(temp, list_of_v) < 0 and f(temp_next, list_of_v) > 0:
                contour_list.append(temp)
    # 取出边界线上点坐标的两个维度
    cx, cy = extract12(contour_list)
    # 画折线图
    plt.plot(cx, cy)
    # 保存图像
    plt.savefig(title)
    plt.show()

