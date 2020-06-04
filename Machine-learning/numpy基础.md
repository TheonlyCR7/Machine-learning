## 查看对应版本号

![image-20200603211238983](img/image-20200603211238983.png)



## `numpy.array` 的方法

创建 `numpy` 类型的数组

```python
# 数组元素必须为同一类型   可以进行相近类型转化
numpyArray = numpy.array([i for i in range(10)])
# 查看存储数据类型
numpyArray.dtype
# int32
```

### `numpy.zeros()` 方法

```python
# 创建一个都为零的数组
numpyArray1 = numpy.zeros(10)
numpyArray1.dtype
# [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
# float64

# 创建一个都为零的二维数组
numpyArray2 = numpy.zeros((3,4))
# [[0. 0. 0. 0.]
#  [0. 0. 0. 0.]
#  [0. 0. 0. 0.]]

# 具体
numpyArray3 = numpy.zeros(shape=(3,4), dtype=int)
# [[0 0 0 0]
#  [0 0 0 0]
#  [0 0 0 0]]

# 创建都为1 的数组  浮点型
numpyArray4 = numpy.ones(10)
# [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]

numpyArray5 = numpy.ones((3, 5))
# [[1. 1. 1. 1. 1.]
#  [1. 1. 1. 1. 1.]
#  [1. 1. 1. 1. 1.]]

# 指定元素  整型
numpyArray6 = numpy.full((3, 4), 11)
# 或者是
numpyArray6 = numpy.full(shape=(3, 4), fill_value=11)
# [[11 11 11 11]
#  [11 11 11 11]
#  [11 11 11 11]]
```



### `arange` 方法

与 range 方法相同

```python
numpyArray7 = numpy.arange(0, 20, 1)
# [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19]
```

不同的是，`arange` 中的步长(第三个参数)可以为浮点数

```python
numpyArray7 = numpy.arange(0, 3, 0.2)
# [0.  0.2 0.4 0.6 0.8 1.  1.2 1.4 1.6 1.8 2.  2.2 2.4 2.6 2.8]
```



### `linspace`  方法

与range不同的是，数组包含结束元素，第三个参数代表数组有几个数，形成等差数组

```python
numpyArray8 = numpy.linspace(0, 20, 5)
# [ 0.  5. 10. 15. 20.]
```



### `random`  方法

生成随机数

```python
numpyArray9 = numpy.random.randint(0, 10)
# 9
# 生成10个随机数, 在0到10之间(前闭后开)
numpyArray9 = numpy.random.randint(0, 10, 10)
numpyArray9 = numpy.random.randint(0, 10, size=10)
# 生成随机矩阵
numpyArray9 = numpy.random.randint(0, 10, size=(3,5))
```



随机种子的使用，相当于给一种随机性加上 id

```python
numpy.random.seed(10)
numpyArray9 = numpy.random.randint(0, 10, 5)
numpy.random.seed(10)
numpyArray10 = numpy.random.randint(0, 10, 5)
# [9 4 0 1 9]
# [9 4 0 1 9]
```



符合正态分布的浮点数

```python
# 默认值为 符合均值为0 方差为1 的正态分布
numpyArray11 = numpy.random.normal()
# 0.28154955449022134

# 符合均值为10 方差为100 的正态分布
numpyArray12 = numpy.random.normal(10, 100)
# 118.53699455631376

# 符合均值为10 方差为100 的正态分布 的矩阵
numpyArray12 = numpy.random.normal(10, 100, (3, 4))
# [[  -0.41346028  -51.32942027  -53.31777518  -97.55511239]
#  [ -69.96572391  -68.84703358 -181.33005977   31.20567573]
#  [-154.90949209   15.78530079  -85.66367579  313.0670341 ]]
```



## 基本属性

`.ndim`  数组的维数

```python
numpyArray12 = numpy.random.normal(10, 100, (3, 4))
print(numpyArray12.ndim)
# 2
```

`.shape` 表示每个维度上的元素个数

```python
# 二维数组
numpyArray12 = numpy.random.normal(10, 100, (3, 4))
print(numpyArray12.shape)
# (3, 4)      第一个维度上有3个元素  第二个维度上有4个元素
```

`.size`  表示元素个数

```python
numpyArray12 = numpy.random.normal(10, 100, (3, 4))
print(numpyArray12.size)
# 12
```



## `numpy.array` 数据访问

对多维数组某个元素的访问

```python
numpyArray12[2, 2]
# 3
# 也支持切片
numpyArray12[0:5]
# [1 2 5 5 1]

numpyArray12[:2, :3]   # 每列前两个元素   每行前三个元素
```

`.reshape`  改变数组的维度（需要另赋给一个新的数组，原数组未变化）

```python
numpyArray12 = numpy.random.randint(0, 20, (3,4))
# [[15 12  4 19]
#  [ 9  4 18 16]
#  [15 19  9 10]]
numpyArray1 = numpyArray12.reshape(4,3)
# [[15 12  4]
#  [19  9  4]
#  [18 16 15]
#  [19  9 10]]
numpyArray2 = numpyArray12.reshape(2, -1)
# 数组为两行  每一行个数自动分配
# [[ 2  9 14 10  4  5]
#  [13 12 18 18 18 14]]
```



## 合并操作

将多个数组合并

默认情况沿着0维进行拼接

### `.concatenate`

```python
# 只能处理维度相同的向量
x = numpy.array([1,2,3])
y = numpy.array([4,5,6])
z = numpy.concatenate([x, y])
# [1 2 3 4 5 6]
w = numpy.concatenate([x, y, z])
# [1 2 3 4 5 6 1 2 3 4 5 6]
```

设定 axis 的值  改变拼接维度

```python
numpyArray = numpy.random.randint(0, 10, (2,3))
numpyArray1 = numpy.random.randint(10, 20, (2,3))
numpyArray2 = numpy.concatenate([numpyArray,numpyArray1])
numpyArray3 = numpy.concatenate([numpyArray,numpyArray1], axis=1)

numpyArray
# [[6 1 0]
#  [6 3 7]]
numpyArray1
# [[18 13 18]
#  [13 19 14]]
numpyArray2
# [[ 6  1  0]
#  [ 6  3  7]
#  [18 13 18]
#  [13 19 14]]
numpyArray3
# [[ 6  1  0 18 13 18]
# [ 6  3  7 13 19 14]]
```

###   `.vstack`

可以智能处理维度不同的向量在**竖直方向上**的合并（前提是列数保持一致）

```python
numpyArray = numpy.random.randint(0,10, (2,2))
numpyArray1 = numpy.random.randint(10,20, (3,2))
numpyArray2 = numpy.vstack([numpyArray1,numpyArray])

numpyArray   
# [[2 4]
#  [4 7]]
numpyArray1
# [[15 12]
#  [18 10]
#  [16 13]]
numpyArray2
# [[15 12]
#  [18 10]
#  [16 13]
#  [ 2  4]
#  [ 4  7]]
```



### `.hstack` 

可以智能处理维度不同的向量在**水平方向上**的合并（前提是行数保持一致）

```python
numpyArray = numpy.random.randint(0,10, (2,3))
numpyArray1 = numpy.random.randint(10,20, (2,2))
numpyArray2 = numpy.hstack([numpyArray1,numpyArray])

numpyArray
# [[1 4 7]
#  [2 1 8]]
numpyArray1
# [[14 17]
#  [17 10]]
numpyArray2
# [[14 17  1  4  7]
#  [17 10  2  1  8]]
```



## 分割操作

### `.split`

```python
numpyArray = numpy.arange(10)
# 将 numpyArray 分成了三部分
numpyArray1, numpyArray2, numpyArray3 = numpy.split(numpyArray, [3,7])

numpyArray
# [0 1 2 3 4 5 6 7 8 9]
numpyArray1
# [0 1 2]
numpyArray2
# [3 4 5 6]
numpyArray3
# [7 8 9]
```

#### 多维数组的分割操作

默认为按行分割

```python
numpyArray4 = numpy.arange(16).reshape(4, 4)
numpyArray5, numpyArray6 = numpy.split(numpyArray4, [2])

numpyArray4
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]
#  [12 13 14 15]]
numpyArray5
# [[0 1 2 3]
#  [4 5 6 7]]
numpyArray6
# [[ 8  9 10 11]
#  [12 13 14 15]]
```

设定 axis 的值  改变分割操作为按列分割

```python
numpyArray4 = numpy.arange(16).reshape(4, 4)
numpyArray5, numpyArray6 = numpy.split(numpyArray4, [2], axis=1)

numpyArray4
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]
#  [12 13 14 15]]
numpyArray5  分割到了左半部分
# [[ 0  1]
#  [ 4  5]
#  [ 8  9]
#  [12 13]]
numpyArray6   分割到了右半部分
# [[ 2  3]
#  [ 6  7]
#  [10 11]
#  [14 15]]
```

对矩阵进行水平分割  按行分割

```python
numpyArray4 = numpy.arange(16).reshape(4, 4)
numpyArray5, numpyArray6 = numpy.vsplit(numpyArray4, [2])

numpyArray4
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]
#  [12 13 14 15]]
numpyArray5
# [[0 1 2 3]
#  [4 5 6 7]]
numpyArray6
# [[ 8  9 10 11]
#  [12 13 14 15]]
```

对矩阵进行竖直分割 按列分割

```python
numpyArray4 = numpy.arange(16).reshape(4, 4)
numpyArray5, numpyArray6 = numpy.hsplit(numpyArray4, [2])

numpyArray4
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]
#  [12 13 14 15]]
numpyArray5
# [[ 0  1]
#  [ 4  5]
#  [ 8  9]
#  [12 13]]
numpyArray6
# [[ 2  3]
#  [ 6  7]
#  [10 11]
#  [14 15]]
```



`numpy.array` 对数组（矩阵）的操作要比python的内置库要快得多

对于array的操作更加接近于数学上对于矩阵的操作

```python
numpyArray4 = numpy.arange(16).reshape(4, 4)
# 对每个元素都乘 2
numpyArray4 * 2
# 每个元素都加 2
numpyArray + 2
# 其他同理  tan sin cos /
# 取log  默认以e为底
numpy.log(numpyArray4)
# 取log  以2为底
numpy.log2(numpyArray4)
# 取 e^x 平方
numpy.exp(x)    # 每个元素都取e^x
# 取平方
numpy.power(3, x)  # 取3 的x 次方  与  3**x 等价
```



## 矩阵运算

若直接调用运算符号，得到的结果只能是对应元素做对应的运算，而非数学意义上的矩阵运算

```PYTHON
A = numpy.arrange(4).reshape(3,3)
B = numpy.arrange(3).reshape(3,3)
print(A)
print(B)
print(A + B)
print(A - B)
print(A * B)
print(A / B)
```

如果想进行数学意义上的矩阵运算，需要调用函数

```PYTHON
# 矩阵乘法
A.dot(B)
# 矩阵转置
A.T
```



### 向量与矩阵的运算

加法：数学上是不可以直接做加法的

但在这里可以直接加（与矩阵的每一行对应元素做加法）

```python
# 向量
v = numpy.array([1,2])
# 矩阵
A = numpy.arange(4).reshape(2,2)

# 矩阵
# [[0 1]
#  [2 3]]
# 向量
# [1 2]
# 结果
# [[1 3]
#  [3 5]]
```

若想让加法在数学上也成立，可以将向量变为矩阵

```python
v = numpy.vstack([v] * A.shape[0])
# 变为矩阵
# [[1 2]
#  [1 2]]
```

然后可以进行正常的加减运算

也可以通过调用 `.tile` 函数 来实现转换

```python
numpy.tile(v, (2, 1))  # 在行方向上堆叠两次   在列方向上堆叠一次
# 结果
# [[1 2]
#  [1 2]]
```



相乘：向量与矩阵中的每个向量元素对应相乘

```PYTHON
v * A
```

数学意义上的相乘： 调用函数 `.dot` 

```python
v.dot(A)
```



## 矩阵的逆

一般的，只有方阵才存在逆矩阵

`.linalg.inv()` 

```python
B = numpy.linalg.inv(A)
# B为 A 的逆矩阵
# 满足 C = A.dot(B)   C为单位矩阵   
```

对于不是方阵来说，可以求它的 “伪逆矩阵”

`.numpy.linalg.pinv(x)`

```python
A = numpy.arange(4).reshape(2,2)
v = numpy.linalg.pinv(A)
B = v.dot(A)

# [[0 1]
#  [2 3]]
# [[-1.50000000e+00  5.00000000e-01]
#  [ 1.00000000e+00  4.21097322e-17]]
原矩阵与 它的伪逆矩阵相乘  
# [[ 1.00000000e+00 -1.66533454e-16]
#  [ 8.42194643e-17  1.00000000e+00]]
# 矩阵B，右对角线上的数近乎为零，可以看作单位矩阵
```



## 聚合操作

