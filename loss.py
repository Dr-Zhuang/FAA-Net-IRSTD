# -*- coding: UTF-8 -*-
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import re
from PIL import Image


x_axix = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80]
UNet = [0.5990, 0.4765, 0.3232, 0.3166, 0.2613, 0.2385, 0.2046, 0.1940,0.1594, 0.1555, 0.1876, 0.2230, 0.2289, 0.2019, 0.2018, 0.1755, 0.1511, 0.1540, 0.1448, 0.1375, 0.1767, 0.2235, 0.1840, 0.1629, 0.1488, 0.1346, 0.1378, 0.1220, 0.1225, 0.1136, 0.1505, 0.1794, 0.1692, 0.1697, 0.1329, 0.1343, 0.1208, 0.1179, 0.1166, 0.1156, 0.1240, 0.1510, 0.1523, 0.1432, 0.1271, 0.1307, 0.1496, 0.1122, 0.1111, 0.1030, 0.1385, 0.1395, 0.1669, 0.1264, 0.1260, 0.1181, 0.1048, 0.1203, 0.1108, 0.1041, 0.1238, 0.1314, 0.1413, 0.1265, 0.1278, 0.1112, 0.1064, 0.1023, 0.1028, 0.0960, 0.1796, 0.1316, 0.1618, 0.1485, 0.1251, 0.1206, 0.1107, 0.1090, 0.1132, 0.1042]
fca = [0.6221, 0.5694, 0.5425, 0.5065, 0.4585, 0.3783, 0.2735, 0.2232,0.1697, 0.1420, 0.1442, 0.1275, 0.1269, 0.1133, 0.1054, 0.1025, 0.1087, 0.0999, 0.0928, 0.0938, 0.0938, 0.0906, 0.0989, 0.0979, 0.0875, 0.0923, 0.0923, 0.0898, 0.0855, 0.0816, 0.0885, 0.0835, 0.0879, 0.0803, 0.0818, 0.0811, 0.0787, 0.0809, 0.0796, 0.1101, 0.0912, 0.0857, 0.0857, 0.0792, 0.0800, 0.0777, 0.0775, 0.0774, 0.0762, 0.0778, 0.0826, 0.0761, 0.0778, 0.0778, 0.0755, 0.0879, 0.0852, 0.0761, 0.0752, 0.0738, 0.0733, 0.0730, 0.0730, 0.0718, 0.0740, 0.0733, 0.0747, 0.0717, 0.0736, 0.0760, 0.0758, 0.0795, 0.0842, 0.0801, 0.0748, 0.0714, 0.0712, 0.0691, 0.0685, 0.0700]# 开始画图
suc = [0.5890, 0.5665, 0.4032, 0.4066, 0.3513, 0.3222, 0.2906, 0.1840, 0.1704, 0.1699, 0.1616, 0.1527, 0.1210, 0.1167, 0.1062, 0.1301, 0.1227, 0.1022, 0.1063, 0.1154, 0.0970, 0.1028, 0.0944, 0.0872, 0.0960, 0.1183, 0.1042, 0.0941, 0.0940, 0.0846, 0.0866, 0.0774, 0.0824, 0.1310, 0.0948, 0.0975, 0.0767, 0.0841, 0.0759, 0.0823, 0.0759, 0.0752, 0.0766, 0.1025, 0.0811, 0.0830, 0.0771, 0.0756, 0.0834, 0.0839, 0.0832, 0.0724, 0.0723, 0.0794, 0.0744, 0.0797, 0.1105, 0.0812, 0.0745, 0.0703, 0.0743, 0.0892, 0.0719, 0.0737, 0.0671, 0.0712, 0.0669, 0.0671, 0.0996, 0.0821, 0.0689, 0.0664, 0.0689, 0.0633, 0.1023, 0.0717, 0.0725, 0.0671, 0.0626, 0.0681]
# sub_axix = filter(lambda x: x % 200 == 0, x_axix)
# plt.axis([0,1,0,1]) ##（0.5，1）x轴的范围， （0,1.08）y轴的范围
# plt.xticks([i * 0.1 for i in range(0, 11)]) ## 显示的x轴刻度值
# plt.yticks([i * 0.1 for i in range(0, 11)])   ## 显示y轴刻度值

# plt.title('Result Analysis')
plt.plot(x_axix, UNet, color='green', label='UNet')
plt.plot(x_axix, suc, color='red', label='TCNet')
plt.plot(x_axix, fca, color='skyblue', label='TCDNet')

plt.legend()  # 显示图例

plt.xlabel('iteration times')
plt.ylabel('loss')
plt.show()
