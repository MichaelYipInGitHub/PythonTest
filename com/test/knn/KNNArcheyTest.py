# coding:utf-8

import numpy as np
import operator
# matplotlib 绘图模块
import matplotlib.pyplot as plt


# from array import array
# from matplotlib.font_manager import FontProperties

# normData 测试数据集的某行，  dataSet 训练数据集 ，labels 训练数据集的类别，k k的值
def classify(normData, dataSet, labels, k):
    # 计算行数
    dataSetSize = dataSet.shape[0]
    #     print ('dataSetSize 长度 =%d'%dataSetSi  ；                  vzvz ze)
    # 当前点到所有点的坐标差值  ,np.tile(x,(y,1)) 复制x 共y行 1列
    diffMat = np.tile(normData, (dataSetSize, 1)) - dataSet
    # 对每个坐标差值平方
    sqDiffMat = diffMat ** 2
    # 对于二维数组 sqDiffMat.sum(axis=0)指 对向量每列求和，sqDiffMat.sum(axis=1)是对向量每行求和,返回一个长度为行数的数组
    # 例如：narr = array([[ 1.,  4.,  6.],
    #                   [ 2.,  5.,  3.]])
    #    narr.sum(axis=1) = array([ 11.,  10.])
    #    narr.sum(axis=0) = array([ 3.,  9.,  9.])
    sqDistances = sqDiffMat.sum(axis=1)
    # 欧式距离 最后开方
    distance = sqDistances ** 0.5
    # x.argsort() 将x中的元素从小到大排序，提取其对应的index 索引，返回数组
    # 例：   tsum = array([ 11.,  10.])    ----  tsum.argsort() = array([1, 0])
    sortedDistIndicies = distance.argsort()
    #     classCount保存的K是魅力类型   V:在K个近邻中某一个类型的次数
    classCount = {}
    for i in range(k):
        # 获取对应的下标的类别
        voteLabel = labels[sortedDistIndicies[i]]
        # 给相同的类别次数计数
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
    # sorted 排序 返回新的list
    #     sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    sortedClassCount = sorted(classCount.items(), key=lambda x: x[1], reverse=True)
    return sortedClassCount[0][0]


def file2matrix(filename):
    fr = open(filename, "rb")
    # readlines:是一次性将这个文本的内容全部加载到内存中(列表)
    arrayOflines = fr.readlines()
    numOfLines = len(arrayOflines)
    #     print "numOfLines = " , numOfLines
    # numpy.zeros 创建给定类型的数组  numOfLines 行 ，3列
    returnMat = np.zeros((numOfLines, 3))
    # 存结果的列表
    classLabelVector = []
    index = 0
    for line in arrayOflines:
        # 去掉一行的头尾空格
        line = line.decode("utf-8").strip()
        listFromline = line.split('\t')
        returnMat[index, :] = listFromline[0:3]
        classLabelVector.append(int(listFromline[-1]))
        index += 1
    return returnMat, classLabelVector

# 将数据归一化
def autoNorm(dataSet):
    #     dataSet.min(0)   代表的是统计这个矩阵中每一列的最小值     返回值是一个矩阵1*3矩阵
    # 例如： numpyarray = array([[1,4,6],
    #                        [2,5,3]])
    #    numpyarray.min(0) = array([1,4,3])    numpyarray.min(1) = array([1,2])
    #    numpyarray.max(0) = array([2,5,6])    numpyarray.max(1) = array([6,5])
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    # dataSet.shape[0] 计算行数， shape[1] 计算列数
    m = dataSet.shape[0]

    #     print '行数 = %d' %(m)
    #     print maxVals

    #     normDataSet存储归一化后的数据
    #     normDataSet = np.zeros(np.shape(dataSet))
    # np.tile(minVals,(m,1)) 在行的方向上重复 minVals m次 即复制m行，在列的方向上重复munVals 1次，即复制1列
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    normDataSet = normDataSet / np.tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


def datingClassM():
    rate = 0.1
    datingDataMat, datingLabels = file2matrix('./archeyTestSet.txt')
    # 将数据归一化
    normMat, ranges, minVals = autoNorm(datingDataMat)
    # m 是 ： normMat行数 = 1000
    m = normMat.shape[0]
    #     print 'm =%d 行'%m
    # 取出100行数据测试
    numTestVecs = int(m * rate)
    errorCount = 0.0
    for i in range(numTestVecs):
        # normMat[i,:] 取出数据的第i行,normMat[numTestVecs:m,:]取出数据中的100行到1000行 作为训练集，
        # datingLabels[numTestVecs:m] 取出数据中100行到1000行的类别，4是K
        classifierResult = classify(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 4)
        print('模型预测值: %d ,真实值 : %d' % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]):
            errorCount += 1.0
    errorRate = errorCount / float(numTestVecs)
    print('正确率 : %f' % (1 - errorRate))
    return 1 - errorRate


def classifyperson():
    resultList = ['输', '平', '赢']
    #你自己的数据
    # input_man = [20000, 10, 2.8]
    input_man = [17934,0.000000,0.147573]
    datingDataMat, datingLabels = file2matrix('archeyTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    result = classify((input_man - minVals) / ranges, normMat, datingLabels, 5)
    print('你这场比赛的预测结果是:%s' % resultList[result - 1])


if __name__ == '__main__':
    #     createScatterDiagram观察数据的分布情况
    #     createScatterDiagram()
    acc = datingClassM()
    if (acc > 0.9):
        classifyperson()
