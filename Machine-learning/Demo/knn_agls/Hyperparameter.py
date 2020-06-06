import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

from knn_agls.KNN_test import train_test_split
from knn_agls.my_own_KNN import KNNClassifier


# 寻找超参数
# 调用手写数字的数据集
from sklearn import datasets

# 调用手写数字的数据集
digits = datasets.load_digits()
digits.keys()
# print(digits.DESCR)

X = digits.data
y = digits.target

some_digit = X[666]
some_digit_image = some_digit.reshape(8, 8)
import matplotlib
plt.imshow(some_digit_image, cmap=matplotlib.cm.binary)
# plt.show()

best_k = 1
best_p = 1
best_score = 0.0
X_train, X_test, y_train, y_test = train_test_split(X, y)
for k in range(1, 11):
    for p in range(1, 6):
        knn_clf = KNeighborsClassifier(n_neighbors=k, p = p)
        knn_clf.fit(X_train, y_train)
        score = knn_clf.score(X_test, y_test)
        if score > best_score:
            best_score = score
            best_k = k
            best_p = p

# 这里的结果是不确定的
print("best_k: ", best_k)
print("best_k: ", best_score)
print("best_p: ", best_p)

# best_k:  9
# best_k:  0.9972144846796658
# best_p:  2





if __name__ == "__main__":
    print()