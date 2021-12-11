import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geohash

colNames = ['mmsi', 'timestamp', 'speed', 'status', 'lon', 'lat', 'draught']
data = pd.read_csv('lng2.csv', delim_whitespace=True, names=colNames)
# 将缺失值0替换为NaN
data['draught'] = data['draught'].replace({0:np.NaN})
# 前填充
data = data.fillna(method='ffill')

# geohash
data['geohash'] = data.apply(lambda x : geohash.encode_uint64(x.lat, x.lon), axis=1)

# print(data.info())
# exit()
dflist = [[]] * 64
# dflist = [pd.DataFrame(columns=colNames)] * 64

for index, row in data.iterrows():
	# print(type(row['geohash']))
	dflist[int(row['geohash']) >> 58].append(row)

for i in range(64):
	dflist[i] = pd.concat(dflist[i], axis=1)
	print(dflist[i].head())
	


# # 输出填充后文件
# data.to_csv('solved.csv', index=False, sep=' ')
# print(data.head())
