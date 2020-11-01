import sys
import json
import time
from pyspark import SparkContext
from operator import add

a = time.time()

sc = SparkContext('local[*]', 'Task1')
sc.setLogLevel("OFF")

run = 'non'

if run == 'local':
    input_path = '../../dataset/Yelp/review.json'
    output_path = '../../dataset/Yelp/result.json'
else:
    input_path = sys.argv[1]
    output_path = sys.argv[2]


def preprocess(line):
    record = json.loads(line)
    user_id = record["user_id"]
    business_id = record["business_id"]
    date = record["date"]
    year = date.split('-')[0].strip()

    return (user_id, business_id, year)


reviews_RDD = sc.textFile(input_path).map(preprocess).repartition(8).persist()

reviews = reviews_RDD.count()
reviews_2018 = reviews_RDD.filter(lambda x: x[2] == "2018").count()
user_temp = reviews_RDD.map(lambda record: (record[0], 1)).reduceByKey(add)
user_count = user_temp.count()
user_top = user_temp.takeOrdered(10, lambda x: (-x[1], x[0]))
business_temp = reviews_RDD.map(lambda record: (record[1], 1)).reduceByKey(add)
business_count = business_temp.count()
business_top = business_temp.takeOrdered(10, lambda x: (-x[1], x[0]))

result_dict = dict()
result_dict["n_review"] = reviews
result_dict["n_review_2018"] = reviews_2018
result_dict["n_user"] = user_count
result_dict["top10_user"] = user_top
result_dict["n_business"] = business_count
result_dict["top10_business"] = business_top

with open(output_path, 'w') as fp:
    json.dump(result_dict, fp)

print(time.time() - a)

if __name__ == '__main__':
    pass