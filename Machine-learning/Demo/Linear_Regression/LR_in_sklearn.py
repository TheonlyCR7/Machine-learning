from sklearn import datasets

# 线性回归的方法在 `scilit-learn` 模块中封装
from sklearn.model_selection import train_test_split

boston = datasets.load_boston()

X = boston.data
y = boston.target

X = X[y < 50.0]
y = y[y < 50.0]


X_train, X_test, y_train, y_test = train_test_split(X, y)

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

if __name__ == "__main__":
    print(lin_reg.coef_)
    print("lin_reg.intercept_: ", lin_reg.intercept_)
    print("R^2: ", lin_reg.score(X_test, y_test))