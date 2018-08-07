from numpy import *
import operator
from os import listdir
import matplotlib.pyplot as plt


def classify(inx,dataset,labels,k):
    datasetSize = dataset.shape[0]
    ### 计算距离
    diffMat = tile(inx,(datasetSize,1)) - dataset
    sqDiffMat = diffMat**2
    sqDistance = sqDiffMat.sum(axis=1)
    distance = sqDistance ** 0.5
    ### 计算距离
    ### 距离从小到大排序
    sortedDistIndicies = distance.argsort()
    ### 字典声明
    classCount = {}
    ### 前k个距离最小
    for i in range(k):
        votelabel = labels[sortedDistIndicies[i]]
        classCount[votelabel] = classCount.get(votelabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0.0,0.0],[0.0,0.1]])
    label = ['A','A','B','B']
    return group,label

'''
def Draw(xs,ys):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xs, ys)
    plt.show()
'''


def firstTest():
    test1 = (1.0, 1.2)
    test2 = (0.0, 0.4)
    dataset, labels = createDataSet()
    conclusion1 = classify(test1, dataset, labels, 3)
    conclusion2 = classify(test2, dataset, labels, 3)
    print(str(test1) + "分类后的结果是属于" + conclusion1 + "类")
    print(str(test2) + "分类后的结果是属于" + conclusion2 + "类")

if __name__ == '__main__':
    # group,label = createDataSet()
    # Draw(group[:,0],group[:,1])
    firstTest()
