import numpy as np
import matplotlib.pyplot as plt


# 实现自己的梯度下降法
# 通过一个二次函数
plot_x = np.linspace(-1., 6., 141)
plot_y = (plot_x-2.5)**2 - 1.   # 二次曲线
plt.plot(plot_x, plot_y)
plt.show()

epsilon = 1e-8
eta = 0.01  # 步长

# 函数
def J(theta):
    return (theta - 2.5) ** 2 - 1.

# 求导
def dJ(theta):
    return 2 * (theta - 2.5)

# x的初始值
theta = 0.0
# 将x的值放到一个数组中
theta_history = [theta]
while True:
    gradient = dJ(theta)
    last_theta = theta
    theta = theta - eta * gradient   # 步长乘以该点的导数
    theta_history.append(theta)

    # 当x点对应的函数值与最小值无限接近时
    if (abs(J(theta) - J(last_theta)) < epsilon):
        break

# 绘制 x 点的运动轨迹
plt.plot(plot_x, plot_y)
plt.plot(np.array(theta_history), J(np.array(theta_history)), color='r', marker='+')


if __name__ == "__main__":
    print(theta)
    print(J(theta))
    print(theta_history)
    plt.show()



