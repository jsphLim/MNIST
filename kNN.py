import os, sys
import numpy as np
import operator
def classify0(inX,dataSet,labels,k):
    dataSetSize=dataSet.shape[0]   #shape读取数据矩阵第一维度的长度
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet  #tile重复数组inX，有dataSet行 1个dataSet列，减法计算差值
    sqDiffMat=diffMat**2 #**是幂运算的意思，这里用的欧式距离
    sqDisttances=sqDiffMat.sum(axis=1) #普通sum默认参数为axis=0为普通相加，axis=1为一行的行向量相加
    distances=sqDisttances**0.5
    sortedDistIndicies=distances.argsort() #argsort返回数值从小到大的索引值（数组索引0,1,2,3）
 #选择距离最小的k个点
    classCount={}
    for i in range(k):
        voteIlabel=labels[sortedDistIndicies[i]] #根据排序结果的索引值返回靠近的前k个标签
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1 #各个标签出现频率
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True) #排序频率
    #python3中：classCount.iteritems()修改为classCount.items()
    #sorted(iterable, cmp=None, key=None, reverse=False) --> new sorted list。
    #reverse默认升序 key关键字排序itemgetter（1）按照第一维度排序(0,1,2,3)
    return sortedClassCount[0][0]  #找出频率最高的

def img2vector(filename):
    returnVect=np.zeros((1,1024))#每个为32x32大小的二进制矩阵 转为1x1024 numpy向量数组
    fr=open(filename)
    for i in range(32):#读出前32行
        lineStr=fr.readline()
        for j in range(32):
            returnVect[0,32*i+j]=int(lineStr[j])#存储在numpy数组中
    return returnVect
#测试
def handwritingClassTest():
    hwLabels=[]
    trainingFileList=os.listdir('digits/trainingDigits')
    m=len(trainingFileList)
    trainingMat=np.zeros((m,1024)) #定义文件数x每个向量的训练集
    for i in range(m):
        fileNameStr=trainingFileList[i]
        fileStr=fileNameStr.split('.')[0]#解析文件
        classNumStr=int(fileStr.split('_')[0])#解析文件名
        hwLabels.append(classNumStr)#存储类别
        trainingMat[i,:]=img2vector('digits/trainingDigits/%s'%fileNameStr) #读入第i个文件内的数据
    #测试数据集
    testFileList=os.listdir('digits/testDigits')
    errorCount=0.0
    mTest=len(testFileList)
    for i in range(mTest):
        fileNameStr=testFileList[i]
        fileStr=fileNameStr.split('.')[0]
        classNumStr=int(fileStr.split('_')[0])#从文件名中分离出数字作为基准
        vectorUnderTest=img2vector('digits/testDigits/%s'%fileNameStr)#访问第i个文件内的测试数据，不存储类 直接测试
        classifierResult=classify0(vectorUnderTest,trainingMat,hwLabels,3)
        print("the classifier came back with: %d,the real answer is: %d" %(classifierResult,classNumStr))
        if(classifierResult!=classNumStr):
            errorCount+=1.0
        print("\nthe total number of errors is: %d" % errorCount)
        print("\nthe total rate is:%f"% (errorCount/float(mTest)))


handwritingClassTest()
