import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geohash

colNames = ["mmsi", "timestamp", "status", "speed", "lon", "lat", "draught"]
data = pd.read_csv("lng2.csv", delim_whitespace=True, names=colNames)
# 将缺失值0替换为NaN
# data["draught"] = data["draught"].replace({0: np.NaN})
# 前填充
# data = data.fillna(method="ffill")

prev, flg = None, 0
# for row in data.index:
for i in data.index:
    if i == 0:
        continue
    if data.at[i, "draught"] == 0 and data.at[i, "mmsi"] == data.at[i - 1, "mmsi"]:
        data.at[i, "draught"] = data.at[i - 1, "draught"]

print("ok1")
# for row in data.itertuples(index=True):
#     if flg == 0:
#         flg += 1
#     elif row.draught == 0 and row.mmsi == prev.mmsi:
#         data.at[row.index, "draught"] = prev.draught
#     prev = row

# geohash
data["geohash"] = data.apply(lambda x: geohash.encode_uint64(x.lat, x.lon), axis=1)
print("ok2")

# # print(data.info())
# # exit()
# dflist = [[]] * 64
# # dflist = [pd.DataFrame(columns=colNames)] * 64

# for index, row in data.iterrows():
# # print(type(row['geohash']))
# dflist[int(row["geohash"]) >> 58].append(row)

# for i in range(64):
# dflist[i] = pd.concat(dflist[i], axis=1)
# print(dflist[i].head())


# # 输出填充后文件
data.drop(data[data["draught"] == 0].index, inplace=True)
print(data.head())
data.to_csv("solved.csv", index=False, sep=" ")
# print(data.head())
