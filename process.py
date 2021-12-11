import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import haversine_distances

def get_centeroid(cluster):
	lon, lat = 0, 0
	for row in cluster.itertuples(index=False):
		lat += row.lat
		lon += row.lon
	return [lat / len(cluster), lon / len(cluster)] # [longitude, latitude]

def _dbscan(df, eps_rad, msp, filter_speed=True):
	# filter the speed
	if filter_speed:
		df = df[df['speed'] <= 1]
	df = df.sample(frac=1 if len(df) <= 48000 else 48000/len(df))
	if len(df) == 0:
		return None

	
	coords = df[['lat', 'lon']]
	db = DBSCAN(eps=eps_rad, min_samples=msp, algorithm='ball_tree', metric='haversine', n_jobs=12).fit(np.radians(coords))
	cluster_lables = db.labels_
	n_clusters = len(set(cluster_lables)) - (1 if -1 in cluster_lables else 0)
	n_noise_ = list(cluster_lables).count(-1)
	if n_clusters == 0:
		return None

	clusters = pd.Series([coords[cluster_lables == n] for n in range(n_clusters)])
	centers = clusters.map(get_centeroid).tolist()
	return centers
	# print(centers)


def closest_node(node, nodes):
	closest_index, dist = -1, 1e12
	for i in range(len(nodes)):
		curdis = haversine_distances(np.radians([node]), np.radians([nodes[i]]))[0][0] * 6371
		if curdis < dist:
			dist, closest_index = curdis, i

	# if dist < 50:
	# 	print("{}\t{},{}\t{},{}".format(dist, node[0], node[1], nodes[closest_index][0], nodes[closest_index][1]))
	return closest_index if dist < 5 else -1

def main():
	data = pd.read_csv('solved.csv', delim_whitespace=True)
	data['geohash'] = pd.to_numeric(data['geohash'])
	print(data.info())

	dflist = [[] for _ in range(256)]
	for row in data.itertuples(index=False):
		dflist[row.geohash >> 56].append(row)

	colNames = ['mmsi', 'timestamp', 'status', 'speed', 'lon', 'lat', 'draught', 'geohash']
	kms_per_radian = 6371.
	# eps_rad = 5 / kms_per_radian
	eps_rad = 2 / kms_per_radian
	centers = []
	for i in range(256):
		df = pd.DataFrame.from_records(dflist[i], columns=colNames)
		# res = _dbscan(df, eps_rad, 500)
		res = _dbscan(df, eps_rad, 500)
		print("---Block {} finished---".format(i))
		if res is None:
			continue
		centers.extend(res)

	centers = pd.DataFrame(centers, columns=['lat', 'lon'])
	result = _dbscan(centers, 10/kms_per_radian, 1, False)
	
	prev, flg = None, 0
	incls, total = 0, 0
	stat = [0 for _ in range(len(result))]
	for row in data.itertuples(index=False):
		if flg == 0:
			flg += 1
		elif abs(row.draught - prev.draught) > 1 and row.mmsi == prev.mmsi:
			# cls = closest_node([data.iloc[i].lat, data.iloc[i].lon], [[-89, 179], [0, 0], [12, 34], [56, 78]])
			cls = closest_node([row.lat, row.lon], result)
			total += 1
			incls += 1 if cls != -1 else 0
			# stat[cls] += 1
			stat[cls] += 1 if row.draught > prev.draught else -1

		prev = row
	print(len(result))
	print(incls / total * 100)
	print(stat)
	lngs = [result[i] for i, e in enumerate(stat) if abs(e) > 2]
	print(len(lngs))
	print(lngs)

	# for i in range(1, len(data)):
	# 	if data.iloc[i].draught != data.iloc[i - 1].draught and data.iloc[i].mmsi == data.iloc[i - 1].mmsi:
	# 		# cls = closest_node([data.iloc[i].lat, data.iloc[i].lon], [[-89, 179], [0, 0], [12, 34], [56, 78]])
	# 		cls = closest_node(data.iloc[i][['lat', 'lon']], result)
	# 		if cls == -1:
	# 			print('class={}, time={}, mmsi={}, loc={},{}'.format(cls, data.iloc[i]['timestamp'], data.iloc[i]['mmsi'], data.iloc[i]['lat'], data.iloc[i]['lon']))
		

if __name__ == '__main__':
	main()