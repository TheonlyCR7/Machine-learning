from time import time

import numpy as np
import matplotlib.pyplot as plt

# 调试梯度
np.random.seed(666)
X = np.random.random(size=(1000, 10))

true_theta = np.arange(1, 12, dtype=float)
X_b = np.hstack([np.ones((len(X), 1)), X])
y = X_b.dot(true_theta) + np.random.normal(size=1000)

# 求出损失函数
def J(theta, X_b, y):
    try:
        return np.sum((y - X_b.dot(theta))**2) / len(X_b)
    except:
        return float('inf')

# 求出对应点的梯度
def dJ_math(theta, X_b, y):
    return X_b.T.dot(X_b.dot(theta) - y) * 2. / len(y)

# 通过debug方式
def dJ_debug(theta, X_b, y, epsilon=0.01):
    res = np.empty(len(theta))
    # 循环求出每个点对应的θ+ θ-
    for i in range(len(theta)):
        theta_1 = theta.copy()
        theta_1[i] += epsilon
        theta_2 = theta.copy()
        theta_2[i] -= epsilon
        # 求出在第i 个维度上导数的值
        res[i] = (J(theta_1, X_b, y) - J(theta_2, X_b, y)) / (2 * epsilon)
    return res

# 梯度下降法
def gradient_descent(dJ, X_b, y, initial_theta, eta, n_iters=1e4, epsilon=1e-8):
    theta = initial_theta
    cur_iter = 0

    while cur_iter < n_iters:
        gradient = dJ(theta, X_b, y)
        last_theta = theta
        theta = theta - eta * gradient
        if (abs(J(theta, X_b, y) - J(last_theta, X_b, y)) < epsilon):
            break

        cur_iter += 1

    return theta


if __name__ == "__main__":
    X_b = np.hstack([np.ones((len(X), 1)), X])
    initial_theta = np.zeros(X_b.shape[1])
    eta = 0.01  # 学习率

    start_time = time()
    theta = gradient_descent(dJ_debug, X_b, y, initial_theta, eta)
    end_time = time()
    print(theta)
    print(start_time - end_time)


