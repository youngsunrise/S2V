import random


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
from cal_struc2vec_similarity import calWCCPA_Similarity





def predModel2019(train_last_index=0, max_node_index=0, u_train=[]):
    # 初步预测模型 M-M+1趋势 应用到 N到N+1
    y_max_sl = u_train[max_node_index]  # M 的值
    y_max_sl_1 = u_train[max_node_index + 1]  # M+1 的值
    y_train_last = u_train[train_last_index]  # 最后一个节点N的值

    y_pred1 = round((y_max_sl_1 - y_max_sl) + y_train_last, 10)
    y_pred2 = round((y_train_last - y_max_sl) / (train_last_index - max_node_index) * (
            train_last_index + 1 - max_node_index) + y_max_sl, 10)

    y_pred2018 = round((y_train_last - y_max_sl) / (train_last_index - max_node_index) + y_train_last, 20)
    # print("y_pred1: ", y_pred1, "y_pred2: ", y_pred2)
    # print("y_pred_2018: ", y_pred2018)

    ### 改进预测，分配节点距离权重
    w1 = float(1) / float(train_last_index + 1 - max_node_index)
    w2 = float(train_last_index - max_node_index) / float(train_last_index + 1 - max_node_index)
    #
    # # 最终预测
    y_pred2019 = w1 * y_pred1 + w2 * y_pred2
    # print("y_2019: ", y_pred)

    return y_pred2018


def gaussianMembership(coss=0):
    weights = []
    down = 0.0000000
    for x in range(1, coss + 1):
        down = down + math.exp(-math.pow((float(x) / float(coss)), 2))
    for x in range(1, coss + 1):
        up = math.exp(-math.pow((float(coss + 1 - x)) / float(coss), 2))
        weights.append(up / down)
    return weights


def normalizedWeights(sim=[]):
    ## 归一化+softmax获取权重
    x_array = np.array(sim)
    array_sum = x_array.sum()
    if array_sum == 0.0:
        x_array = x_array + 1.000000
        x_array = x_array / len(sim)
    else:
        x_array = x_array / array_sum
    return x_array


def getNmaxIndexs(arr=[], topN=0):
    maxIndexs = []
    for i in range(1, topN + 1, 1):
        maxNum = -1000
        indexM = 0
        # 每遍历一次,找到一个最大值
        for index, v in enumerate(arr):
            if v > maxNum:
                maxNum = v
                indexM = index
        maxIndexs.append(indexM)
        # 将以找到的最大值设为极低值
        arr[indexM] = -1000
    return maxIndexs


def getNear(arr=[], topN=0):
    c = 1
    maxIndexs = []
    for i in range(1, topN + 1, 1):
        random_index = len(arr) - c
        maxIndexs.append(random_index)
    return maxIndexs


def oneStepAheadPredict(train_index_end=2, year=2000, topN=3, walk_length=3, num_of_walk=3, datatype="DJI"):
    # datatype = "TAIEX"
    # datatype = "S&P 500"
    # datatype = "HSI"
    # datatype = "DJI"
    # datatype = "SST"

    ## 原数据数据集
    df = pd.read_csv("data\\" + datatype + "\\" + str(year) + "\\allyear" + str(year) + ".csv")
    df_h = df.iloc[:, 1]  #######

    fig, ax = plt.subplots(figsize=(5.5, 5), dpi=300)
    ax.set_xlabel("time", fontsize=14)
    ax.set_ylabel("Highest Index", fontsize=14)
    ax.set_title("" + datatype + " data (year " + str(year) + ")", fontsize=14)
    ax.plot(df_h[:])
    # xxx = range(20,268, 1)
    # ax.plot(xxx, df_h[20:], c="red")
    plt.subplots_adjust(left=0.2, bottom=0.15)
    plt.show()

    # train_index_end = 2  # 训练集截至下标
    dfpreyear = pd.read_csv(
        "data\\" + datatype + "\\" + str(year - 1) + "\\allyear" + str(year - 1) + ".csv")
    df_h2 = dfpreyear.iloc[:, 1]  ######   上一年的所有数据

    # 随后我们对原始数据集进行分割    当年的前四天
    df_train = df_h[:train_index_end]

    frames = [df_h2, df_train]
    df_train = pd.concat(frames)   # delete this when doing ablation study


    print("训练集元素数目", np.shape(df_train))
    train_len = len(df_train)
    u_train = df_train.values.tolist()  # 转换为list，便于操作

    df_test = df_h[train_index_end:]
    print("测试集元素数目", np.shape(df_test))
    test_len = len(df_test)
    v_test = df_test.values.tolist()  # 转换为list，便于操作

    y_pred_result = []
    # 调用我们封装的JSD+局部网络函数
    # 计算节点相似度矩阵simMatrix
    SimMatrix, G = calWCCPA_Similarity(u_train, walk_length, num_of_walk)
    train_last_index = len(u_train) - 1  # 获取训练集最后一个元素下标

    # 寻找到末尾节点相似度最高的节点
    max_sl = -1000
    max_node_index = 0
    # 要寻找多个
    arr = []
    for x in range(0, train_last_index, 1):
        # 将最后一列相似度向量(也就是所有节点到末尾节点的相似度)赋值给arr
        arr.append(SimMatrix[x][train_last_index])

    max_node_indexs = getNmaxIndexs(arr, topN)
    ############################## for ablation study
    # # print("len arr is ", len(arr))
    # max_node_indexs = getNear(arr, topN)
    # print(max_node_indexs)
    ##############################

    max_sl_v = []
    for index in max_node_indexs:
        max_sl_v.append(SimMatrix[index][train_last_index])

    print("与末尾节点", train_last_index + 1,
          "相似度最高的" + str(topN) + "节点下标为：", max_node_indexs, "  \n相似度为：", max_sl_v)

    # 获取topN个预测值
    y_predsteps = []
    for index in max_node_indexs:
        y_predsteps.append(predModel2019(train_last_index, index, u_train))
    # 获取高斯成员函数权重
    # weights = gaussianMembership(len(max_node_indexs))
    # 正则化+softmax获取权重
    weights = normalizedWeights(max_sl_v)
    # 预测模型
    # weights = [0.333333,0.333333,0.333333]
    y_pred = 0.0000000
    for index, v in enumerate(weights):
        y_pred += (v * y_predsteps[index])
    y_pred_result.append(y_pred)  # 将第一步预测的结果加入预测结果集合

    for x in range(1, len(v_test), 1):  # 每预测一个值，就将V中一个实际值加入U，直到U包含所有V的值 ####len(v_test)+1
        u_train.append(v_test[0])

        # 预测后，将v中实际值加入训练集
        if (len(v_test) != 1):
            v_test = v_test[1:]
            # 训练集元素减一
        SimMatrix, G = calWCCPA_Similarity(u_train, walk_length, num_of_walk)
        train_last_index = len(u_train) - 1  # 获取训练集最后一个元素下标

        # 要寻找多个
        arr = []
        for x in range(0, train_last_index, 1):
            # 将最后一列相似度向量(也就是所有节点到末尾节点的相似度)赋值给arr
            arr.append(SimMatrix[x][train_last_index])

        max_node_indexs = getNmaxIndexs(arr, topN)
        ############################## for ablation study
        # # print("len arr is ", len(arr))
        # max_node_indexs = getNear(arr, topN)
        # print(max_node_indexs)
        ##############################

        max_sl_v = []
        for index in max_node_indexs:
            max_sl_v.append(SimMatrix[index][train_last_index])

        print("与末尾节点", train_last_index + 1,
              "相似度最高的" + str(topN) + "节点下标为：", max_node_indexs, "  \n相似度为：", max_sl_v)

        # 获取topN个预测值
        y_predsteps = []
        for index in max_node_indexs:
            y_predsteps.append(predModel2019(train_last_index, index, u_train))
        # 获取高斯成员函数权重
        # weights = gaussianMembership(len(max_node_indexs))
        # print("guassian weights ", weights)
        # 正则化+softmax获取权重
        weights = normalizedWeights(max_sl_v)
        # weights = [0.333333, 0.333333, 0.333333]
        y_pred = 0.0000000
        for index, v in enumerate(weights):
            y_pred += (v * y_predsteps[index])
        y_pred_result.append(y_pred)  # 将第一步预测的结果加入预测结果集合

    print("预测结果集合y_pred_result的len应与初始v的len保持一致")
    print("v初始len: ", len(df_test), "   预测结果集len:", len(y_pred_result))

    y_pred_result = pd.DataFrame(y_pred_result)
    print(y_pred_result)
    y_pred_result.to_csv("data\\" + datatype + "\\1results\\2024S2V\\predicted" + str(year) + "_1.csv")


def multiStepAheadPredict(train_index_end=2, year=2000, topN=3, walk_length=3, num_of_walk=3, L=3, ):
    datatype = "TAIEX"
    # datatype = "S&P 500"
    # datatype = "HSI"
    # datatype = "DJI"
    # datatype = "CCI"

    ## 原数据数据集
    df = pd.read_csv("D:\\codeplace\\data\\" + datatype + "\\" + str(year) + "\\allyear" + str(year) + ".csv")
    df_h = df.iloc[:, 5]  #######
    fig, ax = plt.subplots(figsize=(5.5, 5), dpi=300)
    ax.set_xlabel("time", fontsize=14)
    ax.set_ylabel("Highest Index", fontsize=14)
    ax.set_title("" + datatype + " data (year " + str(year) + ")", fontsize=14)
    ax.plot(df_h[:])
    # xxx = range(20,268, 1)
    # ax.plot(xxx, df_h[20:], c="red")
    plt.subplots_adjust(left=0.2, bottom=0.15)
    plt.show()

    dfpreyear = pd.read_csv(
        "D:\\codeplace\\data\\" + datatype + "\\" + str(year - 1) + "\\allyear" + str(year - 1) + ".csv")
    df_h2 = dfpreyear.iloc[:, 5]  ######

    # 随后我们对原始数据集进行分割
    df_train = df_h[:train_index_end]

    frames = [df_h2, df_train]
    df_train = pd.concat(frames)

    print("训练集元素数目", np.shape(df_train))
    train_len = len(df_train)
    u_train = df_train.values.tolist()  # 转换为list，便于操作

    df_test = df_h[train_index_end:]
    print("测试集元素数目", np.shape(df_test))
    test_len = len(df_test)
    v_test = df_test.values.tolist()  # 转换为list，便于操作

    y_pred_result = []
    # 调用我们封装的JSD+局部网络函数
    # 计算节点相似度矩阵simMatrix
    SimMatrix, G = calWCCPA_Similarity(u_train, walk_length, num_of_walk)
    train_last_index = len(u_train) - 1  # 获取训练集最后一个元素下标

    # 寻找到末尾节点相似度最高的节点
    max_sl = -1000
    max_node_index = 0
    # 要寻找多个
    arr = []
    for x in range(0, train_last_index, 1):
        # 将最后一列相似度向量(也就是所有节点到末尾节点的相似度)赋值给arr
        arr.append(SimMatrix[x][train_last_index])

    max_node_indexs = getNmaxIndexs(arr, topN)
    max_sl_v = []
    for index in max_node_indexs:
        max_sl_v.append(SimMatrix[index][train_last_index])

    print("与末尾节点", train_last_index + 1,
          "相似度最高的" + str(topN) + "节点下标为：", max_node_indexs, "  \n相似度为：", max_sl_v)

    # 获取topN个预测值
    y_predsteps = []
    for index in max_node_indexs:
        y_predsteps.append(predModel2019(train_last_index, index, u_train, G))
    # 获取高斯成员函数权重
    # weights = gaussianMembership(len(max_node_indexs))
    # 正则化+softmax获取权重
    weights = normalizedWeights(max_sl_v)
    # 预测模型

    y_pred = 0.0000000
    for index, v in enumerate(weights):
        y_pred += (v * y_predsteps[index])
    counter = L
    y_pred_result.append(y_pred)  # 将第一步预测的结果加入预测结果集合
    counter -= 1
    u_train_ba = df_train.values.tolist()

    for x in range(1, len(v_test), 1):  # 每预测一个值，就将V中一个实际值加入U，直到U包含所有V的值 ####len(v_test)+1

        u_train_ba.append(v_test[0])
        # if len(u_train) > 90:
        #     u_train = u_train[1:]

        # 预测后，将v中实际值加入训练集
        if (len(v_test) != 1):
            v_test = v_test[1:]
            # 训练集元素减一

        u_train.append(y_pred)
        # print(u_train)
        # print(u_train_ba)
        if counter == 0:
            # 预测了L窗口个值，需要用真实值刷新训练集
            u_train.clear()
            u_train = u_train_ba.copy()
            counter = L
            # reset counter

        SimMatrix, G = calWCCPA_Similarity(u_train, walk_length, num_of_walk)
        # step 参数需要调优 需要满足 π⃗x(tend) − ⃗πx(tend−1) < 10e−5
        train_last_index = len(u_train) - 1  # 获取训练集最后一个元素下标
        # 寻找到末尾节点的局部游走相似度最高的节点
        max_sl = -1000
        max_node_index = 0

        # 要寻找多个
        arr = []
        for x in range(0, train_last_index, 1):
            # 将最后一列相似度向量(也就是所有节点到末尾节点的相似度)赋值给arr
            arr.append(SimMatrix[x][train_last_index])
        # topN = 3
        max_node_indexs = getNmaxIndexs(arr, topN)
        max_sl_v = []
        for index in max_node_indexs:
            max_sl_v.append(SimMatrix[index][train_last_index])

        print("与末尾节点", train_last_index + 1,
              "相似度最高的" + str(topN) + "节点下标为：", max_node_indexs, "  \n相似度为：", max_sl_v)

        # 获取topN个预测值
        y_predsteps = []
        for index in max_node_indexs:
            y_predsteps.append(predModel2019(train_last_index, index, u_train, G))
        # 获取高斯成员函数权重
        # weights = gaussianMembership(len(max_node_indexs))
        # print("guassian weights ", weights)
        # 正则化+softmax获取权重
        weights = normalizedWeights(max_sl_v)
        # 预测模型
        y_pred = 0.0000000
        for index, v in enumerate(weights):
            y_pred += (v * y_predsteps[index])
        y_pred_result.append(y_pred)  # 将第一步预测的结果加入预测结果集合
        counter -= 1

    print("预测结果集合y_pred_result的len应与初始v的len保持一致")
    print("v初始len: ", len(df_test), "   预测结果集len:", len(y_pred_result))

    y_pred_result = pd.DataFrame(y_pred_result)
    print(y_pred_result)
    y_pred_result.to_csv(
        "D:\\codeplace\\data\\" + datatype + "\\1results\\2024S2V\\predicted" + str(year) + "_1(L=" + str(L) + ").csv")


oneStepAheadPredict(train_index_end=4, year=2002, topN=3, datatype="DJI")

#
# for x in range(2002, 2008, 1):
#
#     oneStepAheadPredict(train_index_end=60, year=x, topN=5, walk_length=1, num_of_walk=3, datatype="DJI")
#
# for x in range(2000, 2008, 1):
#
#     oneStepAheadPredict(train_index_end=60, year=x, topN=5, walk_length=1, num_of_walk=3, datatype="HSI")

# oneStepAheadPredict(train_index_end=4, year=2001, topN=3, walk_length=1, num_of_walk=3, datatype="DJI")
## multi-step prediction test
# for x in range(2007, 2008, 1):
#     # multiStepAheadPredict(train_index_end=4, year=x, topN=3, walk_length=1, num_of_walk=3, L=3)
#     multiStepAheadPredict(train_index_end=4, year=x, topN=3, walk_length=1, num_of_walk=3, L=6)
#     multiStepAheadPredict(train_index_end=4, year=x, topN=3, walk_length=1, num_of_walk=3, L=9)
#     # oneStepAheadPredict(train_index_end=4, year=x, topN=3, walk_length=1, num_of_walk=3, datatype="DJI")

# for x in range(2007, 2008, 1):
#     multiStepAheadPredict(train_index_end=4, year=x, topN=3, walk_length=1, num_of_walk=3, L=3)
#     multiStepAheadPredict(train_index_end=4, year=x, topN=3, walk_length=1, num_of_walk=3, L=6)
#     multiStepAheadPredict(train_index_end=4, year=x, topN=3, walk_length=1, num_of_walk=3, L=9)
# oneStepAheadPredict(train_index_end=180, year=2003, topN=3, walk_length=1, num_of_walk=3, datatype="SST")
# oneStepAheadPredict(train_index_end=180, year=2004, topN=3, walk_length=1, num_of_walk=3, datatype="SST")

# for x in range(2004, 2007, 1):
#     oneStepAheadPredict(train_index_end=4, year=x, topN=3, walk_length=1, num_of_walk=3, datatype="HSI")
