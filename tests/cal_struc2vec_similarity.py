import networkx as nx
from numpy import double
from ts2vg import NaturalVG
import numpy as np
import math
from ge import Struc2Vec, Node2Vec, SDNE, LINE, DeepWalk
from numpy.linalg import norm


######################################################################

######################################################################


def distributeWeight(G, e, degreeSum = 0.0000000, ts=[]):
    # 通过VGA聚合算子，为边分配权重
    # e 是一个存储边的tuple
    # print(e)
    # 首先我们需要获取连边对应的两个节点的VGA聚合算子（权重）
    node1, node2 = e[0], e[1]
    # 计算两个节点的VGA算子
    vgaNode1 = round(float(nx.degree(G, node1))/float(degreeSum), 10)
    vgaNode2 = round(float(nx.degree(G, node2))/float(degreeSum), 10)
    # 获取两个节点的权重后，我们为边分配权重，采用平均模式（或者重力模式）
    # 参考：A novel visibility graph transformation of time series into weighted
    # networks (2018)
    ################ 重力模式
    A = 10000 # 至少 100
    Euclideandistance = float(math.pow((node1-node2), 2)) + float(math.pow((ts[node1-1]-ts[node2-1]), 2))
    edgeWeight = round(float(A)*vgaNode1*vgaNode2/Euclideandistance, 10)
    ################

    return edgeWeight

# ts=[20,50,10,30,40]
ts = [0.5, 0.9, 0.6, 0.5, 1.0, 0.75, 0.5, 0.9, 0.76, 0.55, 0.70]

def calljsd(out1, in1, maxrange):
    JSD = 0.0
    double(JSD)
    for x in range(maxrange + 1):
        if out1[x] != 0.0 or in1[x] != 0.0:
            print("out1 and in1:", out1[x], in1[x])
            if (out1[x] == 0):
                kld1 = 0.0
                double(kld1)
            else:
                kld1 = 0.5 * double(out1[x]) * np.log(double(out1[x]) / (double((out1[x] + in1[x])) / 2.0))
            print('kld1:', kld1)
            if (in1[x] == 0):
                kld2 = 0.0
            else:
                kld2 = 0.5 * double(in1[x]) * np.log(in1[x] / ((in1[x] + out1[x]) / 2.0))
            print('kld2:', kld2)
            JSD += (kld1 + kld2)
    # out1 = np.asarray(out1, dtype=np.double)
    # in1 = np.asarray(in1, dtype=np.double)
    # KL = np.sum(np.where(out1 != 0, out1 * np.log(out1 / in1), 0))
    print("计算JSD")
    print(JSD)
    return JSD



###### 构建带权网络
def calWCCPA_Similarity(timeseries=[], walk_length =3, num_of_walk = 3):

    # since we skipped the random walk process in struc2vec, num_of_walk parameter is useless here

    print("time series length: ", len(timeseries))
    # map time series into Visibility Graph
    g = NaturalVG(directed=None).build(timeseries)
    G = g.as_networkx()
    # g2 = NaturalVG(directed=None).build(ts2)
    # G2 = g2.as_networkx()
    # fig, [ax0, ax1, ax2] = plt.subplots(ncols=3, figsize=(12, 3.5))
    #
    # ax0.plot(timeseries)
    # ax0.set_title("Time Series")
    #
    # graph_plot_options = {
    #     "with_labels": False,
    #     "node_size": 2,
    #     "node_color": [(0, 0, 0, 1)],
    #     "edge_color": [(0, 0, 0, 0.15)],
    # }
    #
    # nx.draw_networkx(G, ax=ax1, pos=g.node_positions(), **graph_plot_options)
    # ax1.tick_params(bottom=True, labelbottom=True)
    # ax1.plot(timeseries)
    # ax1.set_title("Visibility Graph")
    #
    # nx.draw_networkx(G, ax=ax2, pos=nx.kamada_kawai_layout(G), **graph_plot_options)
    # ax2.set_title("Visibility Graph "+str(len(timeseries))+"")
    #
    # plt.show()

    nodesNum = G.number_of_nodes()  # 统计结点数 N
    G = nx.relabel_nodes(G, lambda x: str(x + 1))  # 重新标签，使得节点从1开始，可视图建图默认从0
    # G2 = nx.relabel_nodes(G2, lambda x: str(x + 1))  # 重新标签，使得节点从1开始，可视图建图默认从0

    ## struct2vec (good)
    # wl =3 nw = 1 opt3 none  ### walk_length=5, num_walks=3
    # print("================", walk_length, num_of_walk, "===============================")
    # opt3_num_layers = k (scale)
    ## context graph layers = opt3_num_layers !!! 多尺度参数
    model = Struc2Vec(G, walk_length, num_of_walk, workers=12, verbose=40, opt3_num_layers=4)
    # model.train()
    # embeddings = model.get_embeddings()
    layersumProb = model.layersumProb
    print("=========")
    print("layersumProb shape is", np.shape(layersumProb))

    # ## node2vec (bad)
    # model = Node2Vec(G, walk_length=10, num_walks=80, p=0.25, q=4, workers=1)  # init model
    # model.train(window_size=5, iter=3)  # train model
    # embeddings = model.get_embeddings()  # get embedding vectors

    # ## line (mid)
    # model = LINE(G, embedding_size=128, order='second')  # init model,order can be ['first','second','all']
    # model.train(batch_size=1024, epochs=50, verbose=2)  # train model
    # embeddings = model.get_embeddings()  # get embedding vectors

    # ## deep work  (bad)
    # model = DeepWalk(G, walk_length=10, num_walks=80, workers=1)  # init model
    # model.train(window_size=5, iter=3)  # train model
    # embeddings = model.get_embeddings()  # get embedding vectors

    ##### SDNE  (只能cpu运算)
    # model = SDNE(G, hidden_size=[256, 128])  # init model
    # model.train(batch_size=3000, epochs=40, verbose=2)  # train model
    # embeddings = model.get_embeddings()  # get embedding vectors


    # print("节点相似度矩阵(基于MSS): ")
    # print(layersumProb)

    return layersumProb, G


# 测试
# calWCCPA_Similarity(ts)




