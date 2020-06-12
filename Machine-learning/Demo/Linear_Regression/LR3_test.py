import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

# 对多元回归算法进行测试
# 载入波士顿房产数据
boston = datasets.load_boston()

X = boston.data
y = boston.target

X = X[y < 50.0]
y = y[y < 50.0]

X_train, X_test, y_train, y_test = train_test_split(X, y)

from Linear_Regression.LinearRegression3 import LinearRegression3
reg = LinearRegression3()
reg.fit_normal(X_train, y_train)


if __name__ == "__main__":

    print(reg.coef_)    # 系数
    print(reg.intercept_)   # 截距
    print(reg.score(X_test, y_test))   # R^2


