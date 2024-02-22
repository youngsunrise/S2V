import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math

# 平均绝对误差
def calMAD(y = [], y_pred = []):
    T = len(y)
    sum = 0.0
    for index,v in enumerate(y):
        # print(index)
        # print(y_pred[index], v)
        sum += np.absolute(y_pred[index]-v)
    return sum/float(T)
# 平均绝对误差百分比
def calMAPE(y = [], y_pred = []):
    T = len(y)
    sum = 0.0
    for index, v in enumerate(y):
        # print(index)
        # print(y_pred[index], v)
        sum += np.absolute((y_pred[index]-v)/v)*100
    return  sum/float(T)
# 对称平均绝对百分比误差
def calSMAPE(y= [], y_pred = []):
    T = len(y)
    sum = 0.0
    for index, v in enumerate(y):
        # print(index)
        # print(y_pred[index], v)
        sum += np.absolute(y_pred[index]-v)/(y_pred[index]+v)*100
    return sum*2.0/float(T)
# 均方差
def calRMSE(y = [], y_pred = []):
    T = len(y)
    sum = 0.0
    for index, v in enumerate(y):
        # print(index)
        # print(y_pred[index], v)
        sum += pow(np.absolute(y_pred[index]-v), 2)
    return math.sqrt(sum/float(T))
# 数据时间
year = 2002
# 模型时间
YEAR = 2024
split = 4             # 对照开始下标，（与初始训练集数目一致）
L = 3                 # multi-step prediction

# datatype = "TAIEX"
# datatype = "S&P 500"
# datatype = "HSI"
datatype = "DJI"
# datatype = "CCI"

df = pd.read_csv("D:\\codeplace\\data\\"+datatype+"\\"+str(year)+"\\allyear"+str(year)+".csv")
df_h = df.iloc[:,1]
df2 = pd.read_csv("D:\\codeplace\\data\\"+datatype+"\\1results\\"+str(YEAR)+"S2V\\predicted"+str(year)+"_1(L="+str(L)+").csv")
df_h2 = df2.iloc[:,1].tolist()



# df3 = pd.read_csv("D:\\codeplace\\data\\"+datatype+"\\1results\\2018rw+vg\\multipredicted("+str(year)+" L="+str(L)+")_SRW.csv")
# df_h3 = df3.iloc[:,1]
# df4 = pd.read_csv("D:\\codeplace\\data\\"+datatype+"\\1results\\"+str(2019)+"rw+vg\\multipredicted("+str(year)+" L="+str(L)+")_SRW.csv")
# df_h4 = df4.iloc[:,1]
# df5 = pd.read_csv("D:\\codeplace\\data\\"+str(datatype)+"\\1results\\"+str(2022)+"vg+jsd\\multipredicted(L="+str(L)+")"+str(year)+"_1.csv")
# df_h5 = df5.iloc[:,1]



fig, ax = plt.subplots(figsize=(5.5,4.5), dpi = 300)
ax.set_xlabel("time", fontsize=14)
ax.set_ylabel("Value", fontsize=14)
ax.set_title(""+datatype+" (year "+str(year)+") L="+str(L)+"", fontsize=14)
ax.plot(df_h[:], linewidth = 1, label="Actual Value")

x_index = range(split, len(df_h), 1)

ax.plot(x_index, df_h2, linewidth = 0.85, label="proposed", color = 'red')

# ax.plot(x_index, df_h3[:], linewidth = 0.8, label="2018", linestyle='dashed')
# ax.plot(x_index, df_h4[:], linewidth = 0.8, label="2019")
# ax.plot(x_index, df_h5[:], linewidth = 0.8, label="JSD", linestyle='dashed', color='#2c2c2c')



ax.legend()
# plt.subplots_adjust(left=0.2, bottom= 0.15)

axins = ax.inset_axes((0.6, 0.1, 0.36, 0.27))
axins.plot(df_h[:], linewidth = 1, label="Actual Value")
axins.plot(x_index, df_h2, linewidth = 0.8, label="proposed", linestyle='dashed', color = 'red')

# axins.plot(x_index, df_h3[:], linewidth = 0.8, label="2018", linestyle='dashed')
# axins.plot(x_index, df_h4[:], linewidth = 0.8, label="2019")
# axins.plot(x_index, df_h5[:], linewidth = 0.8, label="JSD", linestyle='dashed', color='#2c2c2c')

zone_left = 115
zone_right = 135
# 坐标轴的扩展比例（根据实际数据调整）
x_ratio = 0.4 # x轴显示范围的扩展比例
y_ratio = 0.36 # y轴显示范围的扩展比例
# X轴的显示范围
xlim0 = x_index[zone_left]-(x_index[zone_right]-x_index[zone_left])*x_ratio
xlim1 = x_index[zone_right]+(x_index[zone_right]-x_index[zone_left])*x_ratio
# Y轴的显示范围
y = np.hstack((df_h[zone_left:zone_right], df_h2[zone_left:zone_right]))
ylim0 = np.min(y)-(np.max(y)-np.min(y))*y_ratio
ylim1 = np.max(y)+(np.max(y)-np.min(y))*y_ratio
# 调整子坐标系的显示范围
axins.set_xlim(xlim0, xlim1)
axins.set_ylim(ylim0, ylim1)
# mark_inset(ax, axins, loc1=3, loc2=1, fc="none", ec='k', lw=1)
# ax.set_ylim(15000, 33000)

plt.subplots_adjust(left=0.15, bottom= 0.11, top=0.94, right=0.95)
plt.show()



print("MAD: ", calMAD(df_h[split:], df_h2), "\nMAPE: ", calMAPE(df_h[split:], df_h2),
      "\nSMAPE: ", calSMAPE(df_h[split:], df_h2), "\nRMSE", calRMSE(df_h[split:], df_h2))


