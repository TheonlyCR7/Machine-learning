import numpy as np


# 实现最值归一化
x = np.random.randint(0, 100, size=100)
# 进行数据归一化
x = (x - np.min(x)) / (np.max(x) - np.min(x))


if __name__ == "__main__":
    print(x)






