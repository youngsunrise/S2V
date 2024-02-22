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

df3 = pd.read_csv(
    "data\\" + datatype + "\\1results\\" + str(YEAR) + "S2V\\ablation\\predicted" + str(year) + "_1.csv")
df_h3 = df3.iloc[:, 1].tolist()



fig, ax = plt.subplots(1, 2, figsize=(12, 5.5), dpi=300)

ax[0].spines['top'].set_visible(False)
ax[0].spines['right'].set_visible(False)
ax[0].set_xlabel("Day", fontsize=14)
ax[0].set_ylabel("Opening index", fontsize=14)
ax[0].set_title("(" + datatype + " " + str(year) + ") Predict with top-N MSS nodes", fontsize=14)
ax[0].plot(df_h[:], linewidth=1, label="Actual Value")
x_index = range(split, len(df_h), 1)
ax[0].plot(x_index[:], df_h2[:], linewidth=1, label="MSS", color='red')
ax[0].legend()
ax[0].text(0.5, -0.12, '(a)', transform=ax[0].transAxes,
      fontsize=16, fontweight='bold', va='top')

ax[1].spines['top'].set_visible(False)
ax[1].spines['right'].set_visible(False)
ax[1].set_xlabel("Day", fontsize=14)
ax[1].set_ylabel("Opening index", fontsize=14)
ax[1].set_title("(" + datatype + " " + str(year) + ") Predict with the nearest N nodes", fontsize=14)
ax[1].plot(df_h[:], linewidth=1, label="Actual Value")
x_index = range(split, len(df_h), 1)
ax[1].plot(x_index[:], df_h3[:], linewidth=1, label="nearest ", color='green')
ax[1].legend()
ax[1].text(0.5, -0.12, '(b)', transform=ax[1].transAxes,
      fontsize=16, fontweight='bold', va='top')



plt.subplots_adjust(left=0.15, bottom=0.15, top=0.94, right=0.95)
plt.show()

print("MAD: ", calMAD(df_h[split:], df_h2), "\nMAPE: ", mean_absolute_percentage_error(df_h[split:], df_h2)*100,
      "\nSMAPE: ", smape(df_h[split:], df_h2), "\nRMSE", calRMSE(df_h[split:], df_h2))
print("MAD: ", calMAD(df_h[split:], df_h3), "\nMAPE: ", mean_absolute_percentage_error(df_h[split:], df_h3)*100,
      "\nSMAPE: ", smape(df_h[split:], df_h3), "\nRMSE", calRMSE(df_h[split:], df_h3))
