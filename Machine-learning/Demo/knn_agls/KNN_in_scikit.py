from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# 训练数据    x代表坐标点   y代表类别
raw_data_X = [[3.393533211, 2.331273381],
              [3.110073483, 1.781539638],
              [1.343808831, 3.368360954],
              [3.582294042, 4.679179110],
              [2.280362439, 2.866990263],
              [7.423436942, 4.696522875],
              [5.745051997, 3.533989803],
              [9.172168622, 2.511101045],
              [7.792783481, 3.424088941],
              [7.939820817, 0.791637231]
             ]
raw_data_y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
# 实际数据  判断  x 数据点属于何种类别(x or y)
x = np.array([8.093607318, 3.365731514])

# 放到 numpy的数组中  x 为矩阵   y 为向量
X_train = np.array(raw_data_X)
y_train = np.array(raw_data_y)

# 与数据集距离最近的 n_neighbors=7 比较
kNN_classifier = KNeighborsClassifier(n_neighbors=7)
# 进行模型拟合
kNN_classifier.fit(X_train, y_train)
# 对样本进行处理
X_predict = x.reshape(1, -1)
# 进行样本预测
y_predict = kNN_classifier.predict(X_predict)
print(y_predict[0])

if __name__ == "__main__":
    print()

