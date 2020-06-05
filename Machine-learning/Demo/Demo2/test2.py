import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

iris = datasets.load_iris()
# 取数据集的前两列
x = iris.data[:, :2]
# 绘制散点图
# plt.scatter(x[:,0], x[:,1])
y = iris.target
x = iris.data[:,2:]
plt.scatter(x[y==0,0], x[y==0,1], color="red", marker="o")
plt.scatter(x[y==1,0], x[y==1,1], color="blue", marker="+")
plt.scatter(x[y==2,0], x[y==2,1], color="green", marker="x")
plt.show()
# print(iris.keys())
# # print(iris.DESCR)
# print(iris.data)
# dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename'])
if __name__ == "__main__":
    print()