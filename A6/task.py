import sys
import time
import random
import numpy as np
from functools import reduce
from sklearn.cluster import KMeans
from collections import defaultdict
from sklearn.metrics import normalized_mutual_info_score


class Point:
    def __init__(self, cluster_id, data):
        self.cluster_id = cluster_id
        self.data = np.array(data)
        self.ds = None

    def __repr__(self):
        return '(id:{}, data:{})'.format(self.cluster_id, self.data)


def preprocess(path):
    points = []
    with open(path) as fp:
        for row in fp.readlines():
            data = row.strip().split(',')
            cluster_id = int(data[1])
            data = list(map(float, data[2:]))
            points.append(Point(cluster_id, data))
    return points


def mahalanobis_point_cluster(point_data, N, SUM, SUMSQ):
    std_cluster = (SUMSQ/N - (SUM/N)**2)**0.5

    # other ways of handling: Can skip specific dimension. Covariance matrix
    if 0 in std_cluster:
        std_cluster += 0.0001
    centroid = SUM/N
    distance_array = (point_data - centroid)/std_cluster # (xi-ci)/si
    distance = np.linalg.norm(distance_array)
    dim = len(SUM)
    if distance < 2*dim**0.5:
        return True
    return False


def mahalanobis_cluster_cluster(N_a, SUM_a, SUMSQ_a, N_b, SUM_b, SUMSQ_b):
    centroid_a = SUM_a / N_a
    centroid_b = SUM_b / N_b

    std_a = (SUMSQ_a/N_a - (SUM_a/N_a)**2)**0.5
    std_b = (SUMSQ_b/N_b - (SUM_b/N_b)**2)**0.5

    if 0 in std_a:
        std_a += 0.0001

    if 0 in std_b:
        std_b += 0.0001

    distance_array = (centroid_a - centroid_b) / std_a*std_b
    distance = np.linalg.norm(distance_array)
    dim = len(SUM_a)
    if distance < 2 * dim ** 0.5:
        return True
    return False


def write_result(round_data, index_address_map):
    with open(file_out, 'w') as fp:
        fp.write("The intermediate results:\n")
        for itr, ds_count, cs_cluster_count, cs_count, rs_count in round_data:
            fp.write("Round {}: {},{},{},{}\n".format(itr, ds_count, cs_cluster_count, cs_count, rs_count))
        fp.write("\n")
        fp.write("The clustering results:\n")
        for i in range(len(index_address_map)):
            fp.write("{},{}\n".format(i, index_address_map[i].ds if index_address_map[i].ds else -1))


def process(file_input, k, file_output):
    points = preprocess(file_input)
    # to print at last in actual order
    index_address_map = {}
    for i in range(len(points)):
        index_address_map[i] = points[i]

    random.seed(42)
    random.shuffle(points)
    chunk_size = len(points)//5
    dim = len(points[0].data)
    kdash = k*10
    total_in_ds = 0
    total_in_cs = 0
    round_data = []
    cs_cluster_id = 0

    # Start BFR. Load 20% of data in "memory"
    memory_points = points[:chunk_size]
    memory_X = np.array([point.data for point in memory_points])

    kmeans = KMeans(n_clusters=kdash, random_state=42).fit(memory_X)
    memory_y = kmeans.labels_

    clusters = defaultdict(list)

    # Adding the point node to the cluster id
    for i in range(len(memory_X)):
        clusters[memory_y[i]].append(memory_points[i])

    points_rs = []
    points_not_rs = []

    for key, value in clusters.items():
        if len(value) == 1:
            points_rs.append(value[0])
        else:
            points_not_rs.extend(value)

    # points_rs and points_not_rs contain actual point nodes

    not_rs_X = np.array([point.data for point in points_not_rs])
    kmeans = KMeans(n_clusters=k, random_state=42).fit(not_rs_X)
    not_rs_y = kmeans.labels_

    # Initial DS clusters found here. Now need to get CS from RS. If very few RS, skip this step for now
    DS = {i:{'N':0, 'SUM':np.zeros(dim), 'SUMSQ':np.zeros(dim)} for i in range(k)}

    for i in range(len(not_rs_X)):
        y = not_rs_y[i]
        DS[y]['N'] += 1
        DS[y]['SUM'] += not_rs_X[i]
        DS[y]['SUMSQ'] += not_rs_X[i]**2
        # Point node updated to reflect DS it belongs to. Now no need to keep track of this point in this DS
        points_not_rs[i].ds = y
        total_in_ds += 1

    CS = {}
    CS_stats = {}
    CS_points = defaultdict(list)

    # if points in rs less than total cluster centers kdash, skip for now until more points in rs
    # otherwise try to get cs and rs
    if len(points_rs) >= kdash:
        points_rs_X = np.array([point.data for point in points_rs])
        kmeans = KMeans(n_clusters=kdash, random_state=42).fit(points_rs_X)
        points_rs_y = kmeans.labels_
        cs_cluster_id += kdash

        clusters = defaultdict(list)

        for i in range(len(points_rs_X)):
            clusters[points_rs_y[i]].append(points_rs_X[i])
            # Adding point node to CS cluster node. If later contains only 1 point, will be removed from CS_points
            CS_points[points_rs_y[i]].append(points_rs[i])
            total_in_cs += 1 # overshot RS

        points_rs = []  # points neither in DS nor in CS
        for key, value in clusters.items():
            if len(value) == 1:
                points_rs.append(CS_points[key][0])
                del CS_points[key]
                total_in_cs -= 1 # compensate overshoot
            else:
                CS[key] = value
                CS_stats[key] = {}
                CS_stats[key]['N'] = len(value)
                CS_stats[key]['SUM'] = reduce(lambda x, y: x+y, value) # np array
                CS_stats[key]['SUMSQ'] = reduce(lambda x, y: x + y**2, value, np.zeros(dim))  # np array


    round_data.append([1,total_in_ds, len(CS), total_in_cs, len(points_rs)])

    for itr in range(4):
        start = (itr+1)*chunk_size
        end = (itr+2)*chunk_size if itr<3 else len(points)
        memory_points = points[start:end]
        for point in memory_points:
            # check if inside DS
            nearest_DS = -1
            nearest_DS_distance = sys.maxsize

            # There are k DS cluster centers
            for i in range(k):
                centroid = DS[i]['SUM']/DS[i]['N']
                d = np.linalg.norm(centroid-point.data)
                if d < nearest_DS_distance:
                    nearest_DS_distance = d
                    nearest_DS = i

            is_inside_DS = mahalanobis_point_cluster(point.data, DS[nearest_DS]['N'], DS[nearest_DS]['SUM'],
                                          DS[nearest_DS]['SUMSQ'])
            if is_inside_DS:
                # Point assigned to a DS
                point.ds = nearest_DS
                DS[nearest_DS]['N'] += 1
                DS[nearest_DS]['SUM'] += point.data
                DS[nearest_DS]['SUMSQ'] += point.data**2
                total_in_ds += 1
                continue

            # check if inside CS
            nearest_CS = None
            nearest_CS_distance = sys.maxsize

            for key in CS_stats:
                centroid = CS_stats[key]['SUM'] / CS_stats[key]['N']
                d = np.linalg.norm(centroid - point.data)
                if d < nearest_CS_distance:
                    nearest_CS_distance = d
                    nearest_CS = key
            is_inside_CS = nearest_CS and mahalanobis_point_cluster(point.data, CS_stats[nearest_CS]['N'], CS_stats[nearest_CS]['SUM'],
                                              CS_stats[nearest_CS]['SUMSQ'])

            cs_count = 0
            if is_inside_CS:
                cs_count += 1
                # Point assigned to a DS
                CS_stats[nearest_CS]['N'] += 1
                CS_stats[nearest_CS]['SUM'] += point.data
                CS_stats[nearest_CS]['SUMSQ'] += point.data ** 2
                CS[nearest_CS].append(point.data)
                CS_points[nearest_CS].append(point)
                total_in_cs += 1
                continue

            points_rs.append(point)

        # cluster RS points to get CS clusters and remaining RS points
        if len(points_rs) >= kdash:
            points_rs_X = np.array([point.data for point in points_rs])
            kmeans = KMeans(n_clusters=kdash, random_state=42).fit(points_rs_X)
            points_rs_y = kmeans.labels_
            points_rs_y += cs_cluster_id
            cs_cluster_id += kdash
            clusters = defaultdict(list)

            for i in range(len(points_rs_X)):
                clusters[points_rs_y[i]].append(points_rs_X[i])
                # Adding point node to CS cluster node. If later contains only 1 point, will be removed from CS_points
                CS_points[points_rs_y[i]].append(points_rs[i])
                total_in_cs += 1

            points_rs = []  # points neither in DS nor in CS
            for key, value in clusters.items():
                if len(value) == 1:
                    points_rs.append(CS_points[key][0])
                    del CS_points[key]
                    total_in_cs -= 1
                else:
                    CS[key] = value
                    CS_stats[key] = {}
                    CS_stats[key]['N'] = len(value)
                    CS_stats[key]['SUM'] = reduce(lambda x, y: x + y, value)  # np array
                    CS_stats[key]['SUMSQ'] = reduce(lambda x, y: x + y ** 2, value, np.zeros(dim))  # np array

        # merge CS clusters if possible
        i = 0
        remaining = list(CS.keys())

        while(i<len(remaining)):
            cluster_a = remaining[i]
            key_a = CS_stats[cluster_a]
            for cluster_b in remaining[i+1:]:
                key_b = CS_stats[cluster_b]
                if mahalanobis_cluster_cluster(key_a['N'], key_a['SUM'], key_a['SUMSQ'],
                                               key_b['N'], key_b['SUM'], key_b['SUMSQ']):
                    # merge b into a
                    key_a['N'] += key_b['N']
                    key_a['SUM'] += key_b['SUM']
                    key_a['SUMSQ'] += key_b['SUMSQ']

                    # Add points to first cluster
                    CS[cluster_a].extend(CS[cluster_b])
                    CS_points[cluster_a].extend(CS_points[cluster_b])

                    # delete b from CS
                    del CS[cluster_b], CS_stats[cluster_b], CS_points[cluster_b]
                    remaining.remove(cluster_b)
            i+=1

        if itr < 3:
            round_data.append([itr+2, total_in_ds, len(CS), total_in_cs, len(points_rs)])

    # merge cs clusters into ds that has closest centroid-centroid distance
    for cluster_id in CS:
        nearest_DS_distance = sys.maxsize
        nearest_DS = None
        cs_centroid = CS_stats[cluster_id]['SUM'] / CS_stats[cluster_id]['N']

        for key in DS:
            ds_centroid = DS[key]['SUM'] / DS[key]['N']
            d = np.linalg.norm(ds_centroid - cs_centroid)
            if d < nearest_DS_distance:
                nearest_DS_distance = d
                nearest_DS = key

        # check md. centroid of CS within DS cluster
        CS_key = CS_stats[cluster_id]
        DS_key = DS[nearest_DS]
        is_near = mahalanobis_cluster_cluster(CS_key['N'], CS_key['SUM'], CS_key['SUMSQ'],
                                               DS_key['N'], DS_key['SUM'], DS_key['SUMSQ'])

        if is_near:
            selected_cluster = nearest_DS
        else:
            selected_cluster = -1

        # best ds found. Update cluster_id of each point within CS. DS stats can be updated but no need later
        for point in CS_points[cluster_id]:
            point.ds = selected_cluster
            if is_near:
                total_in_ds += 1
                total_in_cs -= 1

    round_data.append([5, total_in_ds, len(CS), total_in_cs, len(points_rs)])

    labels_true = [index_address_map[i].cluster_id for i in range(len(index_address_map))]
    labels_pred = [index_address_map[i].ds if index_address_map[i].ds else -1 for i in range(len(index_address_map))]
    print("Score: {}".format(normalized_mutual_info_score(labels_true, labels_pred)))

    write_result(round_data, index_address_map)


if __name__ == '__main__':
    time_start = time.time()
    file_in, num_clusters, file_out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    process(file_in, num_clusters, file_out)
    time_total = time.time() - time_start
    print("Duration: {}s".format(time_total))
