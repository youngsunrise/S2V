import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error


# 平均绝对误差
def calMAD(y=[], y_pred=[]):
    T = len(y)
    sum = 0.0
    for index, v in enumerate(y):
        # print(index)
        # print(y_pred[index], v)
        sum += np.absolute(y_pred[index] - v)
    return sum / float(T)


# 平均绝对误差百分比
def calMAPE(y=[], y_pred=[]):
    T = len(y)
    sum = 0.0
    for index, v in enumerate(y):
        # print(index)
        # print(y_pred[index], v)
        sum += np.absolute((y_pred[index] - v) / v) * 100
    return sum / float(T)


# 对称平均绝对百分比误差
def calSMAPE(y=[], y_pred=[]):
    T = len(y)
    sum = 0.0
    for index, v in enumerate(y):
        # print(index)
        # print(y_pred[index], v)
        sum += np.absolute(y_pred[index] - v) / (y_pred[index] + v) * 100
    return sum * 2.0 / float(T)


def smape(y_true, y_pred):
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    smape_score = np.mean(numerator / denominator) * 100

    return smape_score


# 均方差
def calRMSE(y=[], y_pred=[]):
    T = len(y)
    sum = 0.0
    for index, v in enumerate(y):
        # print(index)
        # print(y_pred[index], v)
        sum += pow(np.absolute(y_pred[index] - v), 2)
    return math.sqrt(sum / float(T))


# 数据时间
year = 2002
# 模型时间
YEAR = 2024
split = 4  # 对照开始下标，（与初始训练集数目一致）
# train_set_init = All data from the previous year + 4(split) days of current year

# datatype = "TAIEX"
# datatype = "S&P 500"
# datatype = "HSI"
datatype = "DJI"
# datatype = "CCI"
# datatype = "ONI"


divide = 4

df = pd.read_csv("data\\" + datatype + "\\" + str(year) + "\\allyear" + str(year) + ".csv")
df_h = df.iloc[:, 1]
df2 = pd.read_csv(
    "data\\" + datatype + "\\1results\\" + str(YEAR) + "S2V\\predicted" + str(year) + "_1.csv")
df_h2 = df2.iloc[:, 1].tolist()


# df3 = pd.read_csv("D:\\codeplace\\data\\"+datatype+"\\1results\\"+str(2018)+"rw+vg\\predicted"+str(year)+"_1(SRW).csv")
# df_h3 = df3.iloc[:,1].tolist()
# df4 = pd.read_csv("D:\\codeplace\\data\\"+datatype+"\\1results\\"+str(2019)+"rw+vg\\predicted"+str(year)+"_1(SRW).csv")
# df_h4 = df4.iloc[:,1].tolist()
# df5 = pd.read_csv("D:\\codeplace\\data\\"+datatype+"\\1results\\"+str(2022)+"vg+jsd\\predicted"+str(year)+"_1.csv")
# df_h5 = df5.iloc[:,1].tolist()
# df6 = pd.read_csv("D:\\codeplace\\data\\"+datatype+"\\1results\\ARIMA\\predicted"+str(year)+"_1.csv")
# df_h6 = df6.iloc[:,1].tolist()

df7 = pd.read_csv("data\\" + datatype + "\\1results\\LSTM\\predicted" + str(year) + "_1.csv")
df_h7 = df7.iloc[:, 1].tolist()

fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=300)

ax.set_xlabel("Day", fontsize=14)
ax.set_ylabel("Value", fontsize=14)
ax.set_title("" + datatype + " data (year " + str(year) + ") S2V", fontsize=14)
ax.plot(df_h[:], linewidth=1, label="Actual Value")

x_index = range(split, len(df_h), 1)

ax.plot(x_index[:], df_h2[:], linewidth=1, label="proposed", color='red')
# ax.plot(x_index[:], df_h7[:], linewidth=1, label="LSTM", color='green', alpha=0.8)

# ax.plot(x_index[:], df_h3[:], linewidth = 1, label="2018(RW)", color = 'orange', alpha = 0.8)
# ax.plot(x_index[:], df_h4[:], linewidth = 0.65, label="2019(RW)", color = "black", alpha = 0.65)
# ax.plot(x_index, df_h5[:], linewidth = 0.8, label="2022(JSD)", linestyle='dashed', color='#2c2c2c')

# ax.set_ylim(10000, 34000)
ax.legend()
# plt.subplots_adjust(left=0.2, bottom= 0.15)
####################
# axins = ax.inset_axes((0.6, 0.1, 0.36, 0.27))
# axins.plot(df_h[:], linewidth = 1, label="Actual Value")
# axins.plot(x_index, df_h2, linewidth = 0.85, label="Proposed", linestyle='dashed', color='red')
####################
# axins.plot(x_index, df_h3[:], linewidth = 0.8, label="2018", linestyle='dashed')
# axins.plot(x_index, df_h4[:], linewidth = 0.8, label="2019")
# axins.plot(x_index, df_h5[:], linewidth = 0.8, label="JSD", linestyle='dashed', color='#2c2c2c')

# zone_left = 175
# zone_right = 195

# zone_left = 110   #2003
# zone_right = 130
# zone_left = 185
# zone_right = 205
# # ###############################
# zone_left = 10
# zone_right = 30
# # 坐标轴的扩展比例（根据实际数据调整）
# x_ratio = 0.45 # x轴显示范围的扩展比例
# y_ratio = 0.2 # y轴显示范围的扩展比例
# # X轴的显示范围
# xlim0 = x_index[zone_left]-(x_index[zone_right]-x_index[zone_left])*x_ratio
# xlim1 = x_index[zone_right]+(x_index[zone_right]-x_index[zone_left])*x_ratio
# # Y轴的显示范围
# y = np.hstack((df_h[zone_left:zone_right], df_h2[zone_left:zone_right]))
# ylim0 = np.min(y)-(np.max(y)-np.min(y))*y_ratio
# ylim1 = np.max(y)+(np.max(y)-np.min(y))*y_ratio
# # 调整子坐标系的显示范围
# axins.set_xlim(xlim0, xlim1)
# axins.set_ylim(ylim0, ylim1)
# # mark_inset(ax, axins, loc1=3, loc2=1, fc="none", ec='k', lw=1)
# ####################################
plt.subplots_adjust(left=0.15, bottom=0.11, top=0.94, right=0.95)
plt.show()

print("MAD: ", calMAD(df_h[split:], df_h2), "\nMAPE: ", mean_absolute_percentage_error(df_h[split:], df_h2)*100,
      "\nSMAPE: ", smape(df_h[split:], df_h2), "\nRMSE", calRMSE(df_h[split:], df_h2))
