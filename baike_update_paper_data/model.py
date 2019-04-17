# coding=utf-8

import os, sys, time, re, random
from datetime import datetime
from sklearn import linear_model, svm, ensemble
from sklearn.feature_selection import chi2, f_classif, f_regression, SelectKBest
from sklearn import preprocessing
import ljqpy
import numpy as np

time.clock()

import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42  # 配置语句
matplotlib.rcParams['ps.fonttype'] = 42
plt.style.use('ggplot')

# data = ljqpy.LoadCSV('data/alldata.txt')
# id2data = {x[-1].split('/')[-1]:x for x in data}
# simdata = ljqpy.LoadCSV('data/sims.txt')

dataX = ljqpy.LoadCSV('baike_data_shawn_94873_main.txt')
dataY = ljqpy.LoadCSV('baike_data_shawn_94873_newY.txt')


def TestModel(model):
    global predY
    model.fit(trainX, trainY)  # 先训练
    ptrainY = model.predict(trainX)  # 再预测
    trainloss = ((ptrainY - trainY) ** 2).sum() / trainY.shape[0]  # 在训练集计算均方误差
    predY = model.predict(testX)   # 在测试集上进行预测
    testloss = ((predY - testY) ** 2).sum() / testY.shape[0]  # 计算测试集的均方误差
    print('trainloss=%.5f  testloss=%.5f' % (trainloss, testloss))  # 打印显示


def GetF1():
    thres = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.25, 0.3]
    for thre in thres:
        tp = (((predY > thre) * 1 + (testY >= 0.25) * 1) == 2).sum()  # TP预测答案正确 也就是模型认为它会更新 并且 它确实更新了
        pp = (testY > 0).sum()  # testY是测试集的标签  测试集上真正更新的样本数
        oo = (predY > thre).sum()  # 模型在测试集上的输出的分数  大于阈值的个数--也就是预测会更新的个数
        prec = tp / oo  # 在预测要更新的oo个样本中，只有tp个样本更新了。也就是预测的精度(precision)
        reca = tp / pp  # 在实际更新的pp个样本中，有tp个样本被预测出来了。也就是召回率(recall)
        f1 = 2 * prec * reca / (prec + reca)
        print('thre:%.2f\tprec:%d/%d %.3f\treca:%d/%d %.3f\tf1:%.3f' % (thre, tp, oo, prec, tp, pp, reca, f1))
        # 输出各个阈值对应的f1分数，选取分数最高的那个阈值作为模型的阈值。也就是说，当模型输出的分数大于该阈值时，该样本被预测为会更新。否则，不会被更新。
    pgs = [(x, y) for x, y in zip(predY, testY)]
    pgs.sort()  # 默认按照第一个数据进行排序 也就是按照模型输出的分数进行升序排序
    prcret = []
    alltp = len([x for x in pgs if x[1] >= 0.25])  # alltp是测试集中有更新的样本数
    tp = alltp  # tp是测试集中实际有更新的样本数
    for k, pg in enumerate(pgs):  # enumerate v. 枚举 将可遍历的数据组合为一个索引序列，同时列出数据(pg)和数据下标(k)
        pp, gg = pg  # 元胞赋值，pp为predY，gg为testY
        prec = tp / (len(pgs) - k)
        reca = tp / alltp
        # print(reca, tp, alltp)
        prcret.append((prec, reca))
        if gg >= 0.25: tp -= 1  # 如果当前的样本值有更新，那么tp-1，使得tp一直等于剩余样本中的有更新的样本的个数
    prcret = list(reversed(prcret))   # PR(precision recall)对
    auc = 0
    for ii in range(1, len(prcret)):
        auc += (prcret[ii - 1][0] + prcret[ii][0]) * (prcret[ii][1] - prcret[ii - 1][1]) * 0.5
    print('auc = %.5f' % auc)  # 上面这个公式是干啥用的 SOS！！
    # 答：这是AUC的计算公式，AUC(Area Under Curve)衡量模型结果好坏。
    # 物理意义：任取一个正例和任取一个负例，正例排序在负例之前的概率。(具体的可参考百度..)
    return prcret


def MakeSingleDataset():
    global X, Y, mms
    X, Y = [], []
    for zx, zy in zip(dataX, dataY):
        Y.append(float(zy[1]))
        xx = [int(x) for x in zx[1:6] + zx[7:]]  # 提取x的第一列到第五列，第七列及第七列之后的数据
        xx[0] /= 7    # 将创建天数除以7得到创建的周数
        xx.append(xx[1] / xx[0])  # 总的编辑次数除以编辑周数
        X.append(xx)
    # 经过处理之后，Y里面是 更新次数
    # X的内容
    # |时间T（周数） | 编辑次数 | 浏览次数| 链接到百科词条的链接数|链接数 |正文文本长度|标题长度|平均每周的编辑次数
    # |  0         |   1     |   2   |          3         |  4   |    5     |   6  |      7        |
    # 论文中用的是Main content length 代码中用的是标题长度

    X = np.array(X)
    Y = np.array(Y)
    np.random.seed(1333)
    np.random.shuffle(X)  # 打乱X的样本顺序。该函数的作用类似洗牌，将输入的元素随机排序。
    np.random.seed(1333)
    np.random.shuffle(Y)  # 选取同样的随机数种子，将Y的顺序按照X那样打乱，打乱前后XY的对应关系不变
    X[:, 1:-1] = np.log(X[:, 1:-1] + 1)  # 对应文中的2-7个特征 第2-7个特征正则化
    mms = preprocessing.MinMaxScaler()  # 将X中的属性值 减去其均值 再除以方差
    X = mms.fit_transform(X)    # 得到的结果是，每个属性/每列来说，所有的数据都聚集在0附近，方差为1
    # test
    # print(np.shape(Y)) 94873
    # print(Y.sum()) 19909   19909/94873 = 0.20985
    # test end
    print('y bigest', max(Y))
    Y = Y * 0.25  # 可能也是对Y的一个预处理，但这里为什么是0.25？？？？？？？
    # 解答为什么是0.25，因为模型的输出值是0~1之间的，对应该样本可能更新的概率。还是理解为一个数据归一化处理把


# Y = np.log(Y+1)
# Y = (Y >= 1) * 1

def MakeSimDataset():
    global X, Y
    X, Y = [], []
    for lln in simdata:
        # if not lln[0] in id2data: continue
        z = id2data[lln[0]]
        xx = [int(x) for x in z[3:-1]]
        slst = [x.split(':') for x in lln[1:]][:10]
        avg, sm = None, 0
        for zn, sc in slst:
            # sc = 1
            zz = id2data[zn]
            na = np.array([int(x) for x in zz[3:-1]]) * float(sc)
            if avg is not None:
                avg += na
            else:
                avg = na
            sm += float(sc)
        if sm == 0: continue
        avg /= sm
        xx += [x for x in avg]
        X.append(xx)
        Y.append(float(z[1]))
    X = np.array(X)
    Y = np.array(Y)
    X = np.log(X)


def getEntropy(D):
    """
    Calculate and return entropy of 1-dimensional numpy array D
    """
    length = len(D)
    valueList = list(set(D))  # 该列表中包含了D中的所有值，无重复值。相当于语料库
    numVals = len(valueList)  # 得到该列表的长度，D中总共有numVals个各不相同的取值
    countVals = np.zeros(numVals)  # 创建一个和该列表相同长度的一位数组
    Ent = 0
    for idx, val in enumerate(valueList):  # 第二次见这个函数 idx是数列号 val是列表的数据
        countVals[idx] = len([x for x in D if x == val])
        Ent += countVals[idx] * 1.0 / length * np.log2(length * 1.0 / countVals[idx])
    return Ent


def getMaxInfoGain(D, X, feat=0):
    """
    Calculate maximum information gain w.r.t. the feature which is specified in column feat of the 2-dimensional array X.
    """
    D = np.array(D)
    EntWithoutSplit = getEntropy(D)
    feature = X[:, feat]
    length = len(feature)
    valueList = list(set(feature))
    splits = np.diff(valueList) / 2.0 + valueList[:-1]
    maxGain = 0
    bestSplit = 0
    bestPart1 = []
    bestPart2 = []
    for split in splits:
        Part1idx = np.argwhere(feature <= split)
        Part2idx = np.argwhere(feature > split)
        E1 = getEntropy(D[Part1idx[:, 0]])
        l1 = len(Part1idx)
        E2 = getEntropy(D[Part2idx[:, 0]])
        l2 = len(Part2idx)
        Gain = EntWithoutSplit - (l1 * 1.0 / length * E1 + l2 * 1.0 / length * E2)
        if Gain > maxGain:
            maxGain = Gain
            bestSplit = split
            bestPart1 = Part1idx
            bestPart2 = Part2idx
    return maxGain, bestSplit, bestPart1, bestPart2


featurename = [str(i) for i in range(9)]


def GetInformationGain(X, y):
    # print('Class Labels of entire training dataet: ',y)
    E = getEntropy(y)
    print("Entropy of Class Labels= ", E)
    ret = []
    for col in range(X.shape[1]):
        print('-' * 30)
        print("Best split w.r.t. to feature %s" % featurename[col])
        maxG, bestSplit, Part1, Part2 = getMaxInfoGain(y, X, feat=col)
        print("Maximum Information Gain = ", maxG)
        print("Best Split = ", bestSplit)
        print("Samples in partition 1: ", len(Part1))
        print("Samples in partition 2: ", len(Part2))
        ret.append(maxG)
    return ret


def TryModels():
    global mrf, mridge, prcret, predY

    plt.figure()
    plt.rc('xtick', labelsize=25)
    plt.rc('ytick', labelsize=25)

    plt.xlabel('Recall', fontsize=25)
    plt.ylabel('Precision', fontsize=25)

    predY = testX[:, -1]
    trainloss = ((trainX[:, -1] - trainY) ** 2).sum() / trainY.shape[0]
    testloss = ((predY - testY) ** 2).sum() / testY.shape[0]
    print('trainloss=%.5f  testloss=%.5f' % (trainloss, testloss))
    prcret = GetF1()
    plt.plot([y for x, y in prcret], [x for x, y in prcret], color='b')

    print('ridge')
    mridge = linear_model.RidgeCV(alphas=[0.1, 0.5, 1.0, 5.0, 10.0])
    TestModel(mridge)
    prcret = GetF1()
    print(chi2(trainX, trainY >= 0.25))
    print(mridge.coef_)

    plt.plot([y for x, y in prcret], [x for x, y in prcret], color='g')

    '''
    print('lasso')
    mlasso = linear_model.Lasso(alpha=0.1)
    TestModel(mlasso)
    '''
    '''
    print('linearsvm')
    msvm = svm.LinearSVR()
    TestModel(msvm)
    '''

    print('rf')
    mrf = ensemble.RandomForestRegressor(50, verbose=False)
    TestModel(mrf)
    prcret = GetF1()
    plt.plot([y for x, y in prcret], [x for x, y in prcret], color='r')

    plt.legend(['Baseline', 'Linear', 'Random forest'], loc=1, fontsize=20)


# plt.show()

def LoadPredict():
    add = r'tests\fourth.txt'
    data = ljqpy.LoadCSV(add)
    global XX, predrr
    XX = []
    for z in data:
        xx = [float(x) for x in z[2:]]
        xx[0] /= 7
        xx.append(xx[1] / xx[0])
        XX.append(xx)
    XX = np.array(XX)
    XX[:, 1:-1] = np.log(XX[:, 1:-1] + 1)
    predrr = [(x[0], x[1]) for x in data]


from sklearn.externals import joblib

if __name__ == '__main__':
    MakeSingleDataset()
    # MakeSimDataset()
    spos = int(X.shape[0] * 0.9)
    trainX, testX = X[:spos], X[spos:]
    trainY, testY = Y[:spos], Y[spos:]  # 将数据划分为训练集和测试集 比例9:1
    TryModels()
    GetInformationGain(trainX[:10000], trainY[:10000])
    mrf = ensemble.RandomForestRegressor(100, verbose=False)
    # TestModel(mrf)
    # GetF1()
    # mrf = linear_model.RidgeCV(alphas=[0.1,0.5,1.0,5.0,10.0])
    # mrf.fit(X, Y)
    # joblib.dump(mms, 'mms.pkl', 5)
    # joblib.dump(mrf, 'mrf.pkl', 5)
    # LoadPredict()
    # XX = mms.fit_transform(XX)
    # preds = mrf.predict(XX)
    # predrr = [(x[0], x[1], y) for x,y in zip(predrr, preds)]
    # predrr.sort(key=lambda d:-d[-1])
    # ljqpy.SaveCSV(predrr, 'ret.txt')
    # ljqpy.SaveCSV(predrr, 'ret_linear.txt')
    print('completed %.3f' % time.clock())
