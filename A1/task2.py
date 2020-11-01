import sys
import json
import time
from pyspark import SparkContext
from operator import add

sc = SparkContext('local[*]', 'Task1')
sc.setLogLevel("OFF")

a = time.time()
run = 'non'

if run == 'local':
    input_path = '../../dataset/Yelp/review.json'
    output_path = '../../dataset/Yelp/result.json'
    n_partition = 8
else:
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    n_partition = int(sys.argv[3])

def preprocess(line):
    record = json.loads(line)
    user_id = record["user_id"]
    business_id = record["business_id"]
    date = record["date"]
    year = date.split('-')[0].strip()

    return [user_id, business_id, year]

reviews_RDD = sc.textFile(input_path).map(preprocess).cache()

start_1 = time.time()

business_temp = reviews_RDD.map(lambda record: (record[1], 1)).cache()
partitions_1 = business_temp.getNumPartitions()
partition_size_1 = business_temp.glom().map(len).collect()
business_top = business_temp.reduceByKey(add, 16).takeOrdered(10, lambda x: (-x[1], x[0]))

end_1 = time.time()

business_temp = reviews_RDD.map(lambda record: (record[1], 1)).partitionBy(n_partition, lambda x: (ord(x[0])+ord(x[1]))%n_partition).cache()
partitions_2 = business_temp.getNumPartitions()
partition_size_2 = business_temp.glom().map(len).collect()
business_top = business_temp.reduceByKey(add).takeOrdered(10, lambda x: (-x[1], x[0]))

end_2 = time.time()

result = {}

result['default'] = {}
result['customized'] = {}

result['default']['n_partition'] = partitions_1
result['default']['n_items'] = partition_size_1
result['default']['exe_time'] = end_1 - start_1

result['customized']['n_partition'] = partitions_2
result['customized']['n_items'] = partition_size_2
result['customized']['exe_time'] = end_2 - end_1

with open(output_path, 'w') as fp:
    json.dump(result, fp)

print(time.time()-a)