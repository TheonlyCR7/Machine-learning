# 聚合操作
from time import time
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy

x = numpy.linspace(0, 10, 100)
y = numpy.sin(x)
# 绘制图像
plt.plot(x, y)
cosy = numpy.cos(x)
var = cosy.shape
siny = y.copy()
# plt.plot(x, siny, label="sin(x)")
# plt.plot(x, cosy, color="red", linestyle="--", label="cos(x)")
# plt.axis([-1, 11, -2, 2])
# plt.xlabel("x axis")
# plt.ylabel("y axis")
# plt.legend()
# plt.title("Welcome to the My World!")
plt.scatter(x, siny)
plt.show()


if __name__ == "__main__":
    print(1)





