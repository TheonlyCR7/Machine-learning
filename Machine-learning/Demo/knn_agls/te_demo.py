import numpy as np
import matplotlib.pyplot as plt


# 训练数据    x代表坐标点   y代表类别
raw_data_X = [[3.393533211, 2.331273381],
              [3.110073483, 1.781539638],
              [1.343808831, 3.368360954],
              [3.582294042, 4.679179110],
              [2.280362439, 2.866990263],
              [7.423436942, 4.696522875],
              [5.745051997, 3.533989803],
              [9.172168622, 2.511101045],
              [7.792783481, 3.424088941],
              [7.939820817, 0.791637231]
             ]
raw_data_y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

# 放到 numpy的数组中  x 为矩阵   y 为向量
X_train = np.array(raw_data_X)
y_train = np.array(raw_data_y)
# 描点绘图
plt.scatter(X_train[y_train==0,0], X_train[y_train==0,1], color='g')
plt.scatter(X_train[y_train==1,0], X_train[y_train==1,1], color='r')
# plt.show()
# 实际数据  判断  x 数据点属于何种类别(x or y)
x = np.array([8.093607318, 3.365731514])
# 预测结果
plt.scatter(X_train[y_train==0,0], X_train[y_train==0,1], color='g')
plt.scatter(X_train[y_train==1,0], X_train[y_train==1,1], color='r')
plt.scatter(x[0], x[1], color='b')
# plt.show()

# 计算实际数据与训练数据之间的欧拉距离
from math import sqrt
distances = []
for x_train in X_train:
    d = sqrt(np.sum((x_train - x)**2))
    distances.append(d)

# 将距离排序，索引值放到一个数组中
nearest = np.argsort(distances)

# 记录距离最近的前六个点的索引
k = 6
topK_y = [y_train[neighbor] for neighbor in nearest[:k]]

# 对六个点的类别进行统计
from collections import Counter
votes = Counter(topK_y)
# 返回类别最多的那一类  [(类别, 数量)]
most_votes = votes.most_common(1)


if __name__ == "__main__":
    print(most_votes)
    print("te_demo")

