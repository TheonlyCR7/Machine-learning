
# scikit-learn 中的 SGD 随机梯度下降法
# 只能解决线性模型
from time import time
from sklearn.linear_model import SGDRegressor
from GradientDescent.own_SGD import X_train_standard, X_test_standard, y_train, y_test

# 初始化
sgd_reg = SGDRegressor()
# 进行训练
start_time1 = time()
sgd_reg.fit(X_train_standard, y_train)
end_time1 = time()
sgd_reg.score(X_test_standard, y_test)


if __name__ == "__main__":
    print(end_time1 - start_time1)


