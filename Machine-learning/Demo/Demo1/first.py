import numpy
import matplotlib
import sklearn
import pandas

# numpyArray = numpy.array([i for i in range(10)])
# numpyArray[5] = 3.14
#
# # 创建一个都为零的数组
# numpyArray1 = numpy.zeros(10)
# numpyArray2 = numpy.zeros((3,4))
# numpyArray3 = numpy.zeros(shape=(3,4), dtype=int)
# numpyArray4 = numpy.ones(10)
# numpyArray5 = numpy.ones((3, 5))
# numpyArray6 = numpy.full((3, 4), 11)
# numpyArray7 = numpy.arange(0, 3, 0.2)
# numpyArray8 = numpy.linspace(0, 20, 5)
# numpy.random.seed(10)
# numpyArray9 = numpy.random.randint(0, 10, 5)
# numpy.random.seed(10)
# numpyArray10 = numpy.random.randint(0, 10, 5)

# numpyArray11 = numpy.random.randint(0,10, 10)
# numpyArray12 = numpy.random.randint(0, 20, (3,4))
# numpyArray1 = numpyArray12.reshape(4,3)
# numpyArray2 = numpyArray12.reshape(3, -1)
# x = numpy.array([1,2,3])
# y = numpy.array([4,5,6])
# z = numpy.concatenate([x, y])
# w = numpy.concatenate([x, y, z])
# numpyArray = numpy.random.randint(0, 10, (2,3))
# numpyArray1 = numpy.random.randint(10, 20, (2,3))
# numpyArray2 = numpy.concatenate([numpyArray,numpyArray1])
# numpyArray3 = numpy.concatenate([numpyArray,numpyArray1], axis=1)
# numpyArray = numpy.random.randint(0,10, (2,3))
# numpyArray1 = numpy.random.randint(10,20, (2,2))
# numpyArray2 = numpy.hstack([numpyArray1,numpyArray])
# numpyArray = numpy.arange(10)
# numpyArray1, numpyArray2, numpyArray3 = numpy.split(numpyArray, [3,7])
#
# numpyArray4 = numpy.arange(1,17).reshape(4, 4)
# # numpyArray5, numpyArray6 = numpy.hsplit(numpyArray4, [2])
# numpyArray5 = numpy.log(numpyArray4)
A = numpy.arange(4).reshape(2,2)
# B = numpy.arange(4).reshape(2,2)
# v = numpy.array([1,2])
# # v = numpy.vstack([v] * A.shape[0])
# v = numpy.tile(v, (2, 1))
v = numpy.linalg.pinv(A)
B = v.dot(A)

if __name__ == '__main__':
    print(A)
    print(v)
    print(B)
    # print(v + A)


# [[1 4 7]
#  [2 1 8]]
# [[14 17]
#  [17 10]]
# [[14 17  1  4  7]
#  [17 10  2  1  8]]