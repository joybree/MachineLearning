#-*- coding:utf-8 -*-
#引入numpy库和matplotlib库
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
def test1():
    # 定义等高线图的横纵坐标x，y
    #从左边取值为从 -3 到 3 ，各取5个点，一共取 5*5 = 25 个点
    x = np.linspace(-3, 3, 5)
    y = np.linspace(-3, 3, 5)
    # 将原始数据变成网格数据
    X, Y = np.meshgrid(x, y)

    # 定义等高线高度函数
    def f(x, y):
        return x * y
    print(X.shape)
    print(Y.shape)
    # 填充颜色
    plt.contourf(X, Y, f(X,Y), 4, alpha = 0.2, cmap = plt.cm.hot)
    # 绘制等高线
    C = plt.contour(X, Y, f(X,Y), 4, colors = 'black', linewidth = 0.5)
    # 显示各等高线的数据标签
    plt.clabel(C, inline = True, fontsize = 10)
    plt.show()


def test2():
    love_soccer_prop = 0.65
    total_population = 325*10**6
    num_people_who_love_soccer = int(total_population * love_soccer_prop)
    num_people_who_donot_love_soccer = int(total_population*(1 - love_soccer_prop))
    peo_love_soccer = np.ones(num_people_who_love_soccer)
    peo_donot_love_soccer = np.zeros(num_people_who_donot_love_soccer)
    print(peo_donot_love_soccer)
    all_people = np.hstack([peo_love_soccer, peo_donot_love_soccer])
    print(all_people)
    print(np.mean(all_people))

def test3():
    mu = 0
    sigma = 2
    x = np.arange(-5, 5, 0.1)
    y = stats.norm.pdf(x, mu, sigma)
    plt.plot(x, y)
    plt.show()




if __name__ == "__main__":
    test3()


