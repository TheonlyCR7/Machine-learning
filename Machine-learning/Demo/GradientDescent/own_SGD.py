from time import time

import numpy as np
import matplotlib.pyplot as plt
# 封装我们自己的SGD
from GradientDescent.SimpleLinearRegression import LinearRegression
from knn_agls.KNN_test import train_test_split

m = 100000
x = np.random.normal(size=m)
X = x.reshape(-1,1)
y = 4.*x + 3. + np.random.normal(0, 3, size=m)

lin_reg = LinearRegression()
lin_reg.fit_bgd(X, y)
print(lin_reg.intercept_, lin_reg.coef_)

lin_reg = LinearRegression()
# n_iters=2  使用两倍的样本量
lin_reg.fit_sgd(X, y, n_iters=2)
print(lin_reg.intercept_, lin_reg.coef_)


# 测试准确性
from sklearn import datasets
# 使用波士顿房价数据
boston = datasets.load_boston()
X = boston.data
y = boston.target

X = X[y < 50.0]
y = y[y < 50.0]
# 得出训练数据集 测试数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, seed=666)

# 进行数据归一化
from sklearn.preprocessing import StandardScaler

standardScaler = StandardScaler()
standardScaler.fit(X_train)
# 归一化结果
X_train_standard = standardScaler.transform(X_train)
X_test_standard = standardScaler.transform(X_test)

lin_reg = LinearRegression()
start_time1 = time()
lin_reg.fit_sgd(X_train_standard, y_train, n_iters=2)
end_time1 = time()
R2 = lin_reg.score(X_test_standard, y_test)

if __name__ == "__main__":
    print(end_time1 - start_time1)
    print(R2)









