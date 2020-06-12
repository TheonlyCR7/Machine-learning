import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import datasets

from knn_agls.my_own_KNN import KNNClassifier


# 将算法预测准确度的测试进行了封装
def train_test_split(X, y, test_ratio=0.2, seed=None):
    """将数据 X 和 y 按照test_ratio分割成X_train, X_test, y_train, y_test"""
    assert X.shape[0] == y.shape[0], \
        "the size of X must be equal to the size of y"
    assert 0.0 <= test_ratio <= 1.0, \
        "test_ration must be valid"

    # 若传入参数  则采用参数的随机模式
    if seed:
        np.random.seed(seed)

    shuffled_indexes = np.random.permutation(len(X))

    test_size = int(len(X) * test_ratio)
    test_indexes = shuffled_indexes[:test_size]
    train_indexes = shuffled_indexes[test_size:]

    X_train = X[train_indexes]
    y_train = y[train_indexes]

    X_test = X[test_indexes]
    y_test = y[test_indexes]

    # my_knn_clf = KNNClassifier(k=3)
    # my_knn_clf.fit(X_train, y_train)
    # y_predict = my_knn_clf.predict(X_test)
    #
    # print("KNN数据预测成功个数：" + str(sum(y_predict == y_test)) + " / " + str(len(y_test)))
    # print("KNN数据预测的准确百分比：" + str(sum(y_predict == y_test) / len(y_test)))

    return X_train, X_test, y_train, y_test




if __name__ == "__main__":

    # 调用数据集
    iris = datasets.load_iris()
    iris.keys()
    X = iris.data
    y = iris.target
    shuffled_indexes = np.random.permutation(len(X))
    test_ratio = 0.2
    test_size = int(len(X) * test_ratio)
    test_indexes = shuffled_indexes[:test_size]
    train_indexes = shuffled_indexes[test_size:]
    X_train = X[train_indexes]
    y_train = y[train_indexes]

    X_test = X[test_indexes]
    y_test = y[test_indexes]

    train_test_split(X, y)
    # X_train, X_test, y_train, y_test = train_test_split(X, y)
    #
    # my_knn_clf = KNNClassifier(k=3)
    # my_knn_clf.fit(X_train, y_train)
    # y_predict = my_knn_clf.predict(X_test)

    # print("KNN算法预测成功的数量: " + str(sum(y_predict == y_test)) + " / " + str(len(y_test)))
    # print("KNN算法预测成功百分比: " + str(sum(y_predict == y_test) / len(y_test)))

    # 预测成功的数量: 28 / 30
    # 预测成功百分比: 0.9333333333333333


    # 调用手写数字的数据集
    digits = datasets.load_digits()
    digits.keys()
    # print(digits.DESCR)

    X = digits.data
    y = digits.target

    some_digit = X[666]
    some_digit_image = some_digit.reshape(8, 8)
    plt.imshow(some_digit_image, cmap= matplotlib.cm.binary)
    # plt.show()

    train_test_split(X, y)
    # X_train, X_test, y_train, y_test = train_test_split(X, y)
    # my_knn_clf = KNNClassifier(k=3)
    # my_knn_clf.fit(X_train, y_train)
    # y_predict = my_knn_clf.predict(X_test)

    # print("手写数字预测成功个数：" + str(sum(y_predict == y_test)) + " / " + str(len(y_test)))
    # print("手写数据预测的准确百分比：" + str(sum(y_predict == y_test) / len(y_test)))

    # 手写数字预测成功个数：352 / 359
    # 手写数据预测的准确百分比：0.9805013927576601
