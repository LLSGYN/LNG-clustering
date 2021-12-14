import numpy as np
import pandas as pd
import json, timeit, geohash

from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import haversine_distances


def get_centeroid(cluster):
    lon, lat = 0, 0
    for row in cluster.itertuples(index=False):
        lat += row.lat
        lon += row.lon
    return [lat / len(cluster), lon / len(cluster)]  # [longitude, latitude]


def _dbscan(df, eps_rad, msp, filter_speed=True):
    # filter the speed
    if filter_speed:
        df = df[df["speed"] <= 1]
    
    df = df.sample(frac=1 if len(df) <= 36000 else 36000 / len(df))
    if len(df) == 0:
        return None

    coords = df[["lat", "lon"]]
    db = DBSCAN(
        eps=eps_rad,
        min_samples=msp,
        algorithm="ball_tree",
        metric="haversine",
        n_jobs=12,
    ).fit(np.radians(coords))
    cluster_lables = db.labels_
    n_clusters = len(set(cluster_lables)) - (1 if -1 in cluster_lables else 0)
    if n_clusters == 0:
        return None

    clusters = pd.Series([coords[cluster_lables == n] for n in range(n_clusters)])
    centers = clusters.map(get_centeroid).tolist()
    return centers


def closest_node(node, nodes):
    closest_index, dist = -1, 1e12
    for i in range(len(nodes)):
        curdis = (
            haversine_distances(np.radians([node]), np.radians([nodes[i]]))[0][0] * 6371
        )
        if curdis < dist:
            dist, closest_index = curdis, i

    return closest_index if dist < 5 else -1


start = timeit.default_timer()

colNames = [
    "mmsi",
    "timestamp",
    "status",
    "speed",
    "lon",
    "lat",
    "draught",
    "geohash"
]
data = pd.read_csv("lng2.csv", delim_whitespace=True, names=colNames)

# fill empty entries
prev, flg = None, 0
for i in data.index:
    if i == 0:
        continue
    if data.at[i, "draught"] == 0 and data.at[i, "mmsi"] == data.at[i - 1, "mmsi"]:
        data.at[i, "draught"] = data.at[i - 1, "draught"]

data.drop(data[data["draught"] == 0].index, inplace=True)

# geohash
data["geohash"] = data.apply(lambda x: geohash.encode_uint64(x.lat, x.lon), axis=1)
dflist = [[] for _ in range(1024)]
for row in data.itertuples(index=False):
    dflist[row.geohash >> 54].append(row)

kms_per_radian = 6371.0
eps_rad = 1 / kms_per_radian
centers = []
for i in range(1024):
    df = pd.DataFrame.from_records(dflist[i], columns=colNames)
    res = _dbscan(df, eps_rad, 300)
    if res is None:
        continue
    centers.extend(res)

centers = pd.DataFrame(centers, columns=["lat", "lon"])
result = _dbscan(centers, 2 / kms_per_radian, 1, False)

prev, flg = None, 0
incls, total = 0, 0
stat = [0 for _ in range(len(result))]
for row in data.itertuples(index=False):
    if flg == 0:
        flg += 1
    elif abs(row.draught - prev.draught) > 1 and row.mmsi == prev.mmsi:
        cls = closest_node([row.lat, row.lon], result)
        total += 1
        incls += 1 if cls != -1 else 0
        stat[cls] += 1 if row.draught > prev.draught else -1
    prev = row

lngs = [result[i] for i, e in enumerate(stat) if abs(e) > 3]
ancr_candid = [result[i] for i, e in enumerate(stat) if abs(e) <= 3]
ancr_candid = pd.DataFrame(ancr_candid, columns=["lat", "lon"])
ancrs = _dbscan(ancr_candid, 10 / kms_per_radian, 1, False)

with open("lng_results_list.json", "w") as f:
    i, cnt = -1, 1
    f.write("[\n")
    for point in result:
        i += 1
        if abs(stat[i]) <= 3:
            continue
        if cnt == 1:
            f.write("\t")
        else:
            f.write(",\n\t")
        isIn = True if stat[i] < 0 else False
        rec = {
            "code": cnt,
            "latitude": point[0],
            "longtitude": point[1],
            "isLNG": True,
            "IN": isIn,
        }
        cnt += 1
        json.dump(rec, f)
    for point in ancrs:
        f.write(",\n\t")
        rec = {
            "code": cnt,
            "latitude": point[0],
            "longtitude": point[1],
            "isLNG": False,
            "IN": None,
        }
        cnt += 1
        json.dump(rec, f)
    f.write("\n]")

stop = timeit.default_timer()
print("total time: {} seconds....".format(stop - start))