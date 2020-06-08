import numpy as np
import matplotlib.pyplot as plt


# 线性回归算法的简单实现

x = np.array([1., 2., 3., 4., 5.])
y = np.array([1., 3., 2., 3., 5.])

# plt.scatter(x, y)
# plt.axis([0, 6, 0, 6])
# plt.show()

# 计算 x y 的均值
x_mean = np.mean(x)
y_mean = np.mean(y)
num = 0.0
d = 0.0
for x_i, y_i in zip(x, y):
    # 分子
    num += (x_i - x_mean) * (y_i - y_mean)
    # 分母
    d += (x_i - x_mean) ** 2

a = num/d
b = y_mean - a * x_mean
# 回归表达式
y_hat = a * x + b

# 绘制图像
plt.scatter(x, y)
plt.plot(x, y_hat, color='r')
plt.axis([0, 6, 0, 6])
plt.show()

# 测试
x_predict = 6
y_predict = a * x_predict + b
print(y_predict)

if __name__ == "__main__":
    print()

