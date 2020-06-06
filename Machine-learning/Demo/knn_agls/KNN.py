import numpy as np
from math import sqrt
from collections import Counter


def kNN_classify(k, X_train, y_train, x):

    # K 必须是合法的
    assert 1 <= k <= X_train.shape[0], "k must be valid"
    # x y 训练数据集 必须元素相同  x坐标对应一个y类别
    assert X_train.shape[0] == y_train.shape[0], \
        "the size of X_train must equal to the size of y_train"
    # 实际数据必须具有所有特征值
    assert X_train.shape[1] == x.shape[0], \
        "the feature number of x must be equal to X_train"

    distances = [sqrt(np.sum((x_train - x)**2)) for x_train in X_train]
    nearest = np.argsort(distances)

    topK_y = [y_train[i] for i in nearest[:k]]
    votes = Counter(topK_y)

    return votes.most_common(1)[0][0]