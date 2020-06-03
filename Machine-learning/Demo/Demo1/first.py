import numpy
import matplotlib
import sklearn
import pandas

numpyArray = numpy.array([i for i in range(10)])
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

numpyArray11 = numpy.random.normal()
numpyArray12 = numpy.random.normal(10, 100, (3, 4))

if __name__ == '__main__':
    # print(numpy.__version__)
    # print(numpyArray)
    # print(numpyArray.dtype)
    # print(numpyArray1)
    # print(numpyArray1.dtype)
    # print(numpyArray9)
    print(numpyArray12)