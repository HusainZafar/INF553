import sys
import time
from graphframes import *
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql import functions as F

sc = SparkContext('local[*]', 'Task1')
sc.setLogLevel("OFF")
sqlContext = SQLContext(sc)


def get_input():
    run = 'non'
    if run == 'local':
        ip = 'power_input.txt'
        op = 'output.txt'
    else:
        ip = sys.argv[1]
        op = sys.argv[2]
    return ip, op


def preprocess(path):
    _vertices = set()
    _edges = []
    with open(path) as fp:
        for line in fp.readlines():
            src, dst = line.split()
            _vertices.add(src)
            _vertices.add(dst)
            _edges.append([src, dst])
            _edges.append([dst, src])
    return list(_vertices), _edges


def process(input, output):
    vertices, edges = preprocess(input)
    vertices = sqlContext.createDataFrame([[v] for v in vertices], ["id"])
    edges = sqlContext.createDataFrame(edges, ["src", "dst"])
    g = GraphFrame(vertices, edges)

    result = g.labelPropagation(maxIter=5)
    # result.show()
    communities = result.groupBy("label").agg(F.collect_set("id").alias("community")).select("community").collect()

    result = []
    for row in communities:
        result.append(sorted(row.community))
    result = sorted(result, key=lambda x: (len(x), x))
    with open(output, 'w') as fp:
        for row in result:
            fp.write("'" + "', '".join(row) + "'\n")


if __name__ == '__main__':
    time_start = time.time()
    input_file_path, output_file_path = get_input()
    process(input_file_path, output_file_path)
    time_end = time.time()
    print("Duration {}".format(time_end-time_start))