import numpy

from knn_agls.my_own_KNN import KNNClassifier
from knn_agls.te_demo import X_train, y_train

x = numpy.array([8.093607318, 3.365731514])
X_predict = x.reshape(1, -1)

knn_test = KNNClassifier(k=7)
knn_test.fit(X_train, y_train)
y_predict = knn_test.predict(X_predict)

if __name__ == "__main__":
    print(y_predict[0])