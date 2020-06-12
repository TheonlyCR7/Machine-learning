# 用knn算法进行R^2 测试
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

boston = datasets.load_boston()

X = boston.data
y = boston.target

X = X[y < 50.0]
y = y[y < 50.0]


X_train, X_test, y_train, y_test = train_test_split(X, y)

standardScaler = StandardScaler()
standardScaler.fit(X_train, y_train)
X_train_standard = standardScaler.transform(X_train)
X_test_standard = standardScaler.transform(X_test)

knn_reg1 = KNeighborsRegressor()
knn_reg1.fit(X_train_standard, y_train)


# 进行网格搜索  找到最好的超参数
from sklearn.model_selection import GridSearchCV

param_grid = [
    {
        "weights": ["uniform"],
        "n_neighbors": [i for i in range(1, 11)]     # 取前十行
    },
    {
        "weights": ["distance"],
        "n_neighbors": [i for i in range(1, 11)],
        "p": [i for i in range(1,6)]
    }
]

knn_reg2 = KNeighborsRegressor()
                        #  回归器     超参数数组    并行处理核   输出内容
grid_search = GridSearchCV(knn_reg2, param_grid, n_jobs=-1, verbose=1)
grid_search.fit(X_train_standard, y_train)


if __name__ == "__main__":
    print("KNN算法: :", knn_reg1.score(X_test_standard, y_test))
    print("最好的超参数: ", grid_search.best_params_)
    # 最好的超参数:  {'n_neighbors': 6, 'p': 1, 'weights': 'distance'}
    # k = 6 p = 1 模式为distance 为计算点与点之间的全职

    # 基于网格搜索得到的超参数进行R^2测试
    print(grid_search.best_estimator_.score(X_test, y_test))