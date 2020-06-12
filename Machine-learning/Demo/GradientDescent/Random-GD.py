# 随机梯度下降法
from time import time

import numpy as np
import matplotlib.pyplot as plt

m = 100000

x = np.random.normal(size=m)
X = x.reshape(-1,1)
y = 4.*x + 3. + np.random.normal(0, 3, size=m)

plt.scatter(x, y)
plt.show()

# 普通的梯度下降法
def J(theta, X_b, y):
    try:
        return np.sum((y - X_b.dot(theta)) ** 2) / len(y)
    except:
        return float('inf')


def dJ(theta, X_b, y):
    return X_b.T.dot(X_b.dot(theta) - y) * 2. / len(y)


def gradient_descent(X_b, y, initial_theta, eta, n_iters=1e4, epsilon=1e-8):
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

start_time1 = time()
X_b = np.hstack([np.ones((len(X), 1)), X])
initial_theta = np.zeros(X_b.shape[1])
eta = 0.01
theta = gradient_descent(X_b, y, initial_theta, eta)
end_time1 = time()

# 随机梯度下降法
def dJ_sgd(theta, X_b_i, y_i):
    return 2 * X_b_i.T.dot(X_b_i.dot(theta) - y_i)

def sgd(X_b, y, initial_theta, n_iters):

    t0, t1 = 5, 50
    def learning_rate(t):
        return t0 / (t + t1)

    theta = initial_theta
    for cur_iter in range(n_iters):
        rand_i = np.random.randint(len(X_b))  # 在每一行随机选择一个样本
        gradient = dJ_sgd(theta, X_b[rand_i], y[rand_i])
        theta = theta - learning_rate(cur_iter) * gradient

    return theta

start_time2 = time()
X_b = np.hstack([np.ones((len(X), 1)), X])
initial_theta = np.zeros(X_b.shape[1])
theta1 = sgd(X_b, y, initial_theta, n_iters=m//3)  # 循环次数是样本数的三分之一
end_time2 = time()


if __name__ == "__main__":
    print(theta)
    print("批量梯度下降法: ", end_time1 - start_time1)
    print(theta1)
    print("随机梯度下降法: ", end_time2 - start_time2)

    # 在保证了准确度的前提下 时间效率提高了
    # [2.98957034 4.00600674]
    # 批量梯度下降法: 0.9315066337585449
    # [3.005949  4.0165338]
    # 随机梯度下降法: 0.27825498580932617


