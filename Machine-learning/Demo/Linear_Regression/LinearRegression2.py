from datetime import time

import numpy as np


# 封装自己实现的算法
# 通过向量化运算来优化  主要修改fit函数
class SimpleLinearRegression2:

    def __init__(self):
        """初始化Simple Linear Regression 模型"""
        # 算法计算出的结果变量，方便用户查看  xxxx_
        self.a_ = None
        self.b_ = None

    def fit(self, x_train, y_train):
        """根据训练数据集x_train,y_train训练Simple Linear Regression模型"""
        assert x_train.ndim == 1, \
            "Simple Linear Regressor can only solve single feature training data."
        assert len(x_train) == len(y_train), \
            "the size of x_train must be equal to the size of y_train"

        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)

        num = (x_train - x_mean).dot(y_train - y_mean)
        d = (x_train - x_mean).dot(x_train - x_mean)

        for x, y in zip(x_train, y_train):
            num += (x - x_mean) * (y - y_mean)
            d += (x - x_mean) ** 2

        self.a_ = num / d
        self.b_ = y_mean - self.a_ * x_mean

        return self

    def predict(self, x_predict):
        """给定待预测数据集(向量)x_predict，返回表示x_predict的结果向量"""
        assert x_predict.ndim == 1, \
            "Simple Linear Regressor can only solve single feature training data."
        assert self.a_ is not None and self.b_ is not None, \
            "must fit before predict!"

        # 返回预测的结果向量
        return np.array([self._predict(x) for x in x_predict])

    # 给一个数据
    def _predict(self, x_single):
        """给定单个待预测数据x，返回x的预测结果值"""
        return self.a_ * x_single + self.b_

    # 类输出方法
    def __repr__(self):
        return "SimpleLinearRegression2()"


if __name__ == "__main__":
    m = 10000
    big_x = np.random.randint(m)
    big_y = big_x * 2 + 3 + np.random.normal(size=m)
    start_time1 = time.time()
