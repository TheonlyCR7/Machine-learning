import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

# 对衡量标准不同的处理
# 均方误差
# 平均绝对误差

# 调用波士顿的房产数据
from sklearn.model_selection import train_test_split

from Linear_Regression.LinearRegression2 import SimpleLinearRegression2

boston = datasets.load_boston()
# dict_keys(['data', 'target', 'feature_names', 'DESCR'])

# 只使用房间数量这个特征
x = boston.data[:, 5]
y = boston.target
# 对 x,y 进行限制
x = x[y < 50.0]
y = y[y < 50.0]

# 获取算法预测准确度
x_train, x_test, y_train, y_test = train_test_split(x, y)
reg = SimpleLinearRegression2()
reg.fit(x_train, y_train)

# 绘制图像
plt.scatter(x_train, y_train)
plt.scatter(x_test, y_test, color="c")
plt.plot(x_train, reg.predict(x_train), color='r')

# 误差总和的测试
# 进行MSE的测试
y_predict = reg.predict(x_test)
mse_test = np.sum((y_predict - y_test) ** 2) / len(y_test)

# RMSE  开平方
from math import sqrt

rmse_test = sqrt(mse_test)

# MAE
mae_test = np.sum(np.absolute(y_predict - y_test)) / len(y_test)


if __name__ == "__main__":
    plt.show()
    print("MSE: ", mse_test)
    print("RMSE: ", rmse_test)
    print("MAE: ", mae_test)

    # RMSE 与 MAE 相比 RMSE 要大一些, 因为受到了极大值的影响
    # MSE: 43.131425193207505
    # RMSE: 6.567451955911631
    # MAE: 4.700218707747944
