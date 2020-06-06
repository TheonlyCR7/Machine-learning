from sklearn.neighbors import KNeighborsClassifier

from knn_agls.Hyperparameter import X_train, y_train

# 创建分类器数组
param_grid = [
    {
        'weights': ['uniform'],
        'n_neighbors': [i for i in range(1, 11)]
    },
    {
        'weights': ['distance'],
        'n_neighbors': [i for i in range(1, 11)],
        'p': [i for i in range(1, 6)]
    }
]

knn_clf = KNeighborsClassifier()
from sklearn.model_selection import GridSearchCV

# 定义网格搜索对象
grid_search = GridSearchCV(knn_clf, param_grid)
# 进行模型拟合
grid_search.fit(X_train, y_train)
# 查找最佳参数
print(grid_search.best_estimator_)
# 搜索准确度
print(grid_search.best_score_)
# 对于数组而言，最佳参数
print(grid_search.best_params_)

# 参数 n_jobs 决定了网格搜索时，主机分配多少核进行分类处理操作   -1 为全部核
# 在搜索过程中，进行一些输出
grid_search = GridSearchCV(knn_clf, param_grid, n_jobs=-1, verbose=2)

grid_search.fit(X_train, y_train)

# [Parallel(n_jobs=-1)]: Done 300 out of 300 | elapsed:   28.6s finished

if __name__ == "__main__":
    print()

