import sys
import shutil
import json
import time
import os
from pyspark import SparkContext
from operator import add

sc = SparkContext('local[16]', 'Task3')
sc.setLogLevel("OFF")

run = 'non'
start = time.time()

if run == 'local':
    review_path = '../../dataset/Yelp/review.json'
    business_path = '../../dataset/Yelp/business.json'
    

    output_1 = '../../dataset/Yelp/3.1.json'
    output_2 = '../../dataset/Yelp/3.2.json'
else:
    review_path = sys.argv[1]
    business_path = sys.argv[2]

    output_1 = sys.argv[3]
    output_2 = sys.argv[4]

'''
Review: business_id, stars

Business: business_id, city
'''


def preprocess_review(x):
    data = json.loads(x)
    return [data["business_id"], (data["stars"], 1)]


def preprocess_business(x):
    data = json.loads(x)
    return [data["business_id"], data["city"]]

def partitionCtr(nums):
    sumCount = [0, 0]
    for num in nums:
        sumCount[0] += num
        sumCount[1] += 1
        return [sumCount]

start = time.time()

reviews_RDD = sc.textFile(review_path).map(preprocess_review).partitionBy(8, lambda x: (ord(x[0])+ord(x[1]))%8).reduceByKey(lambda x,y: (x[0]+y[0], x[1]+y[1])).cache()
business_RDD = sc.textFile(business_path).map(preprocess_business).cache()

joined_rdd = reviews_RDD.join(business_RDD).map(lambda x: x[1][::-1]).reduceByKey(lambda x,y: (x[0]+y[0], x[1]+y[1]))\
    .map(lambda x: (x[0], x[1][0]/x[1][1]))

def sort_python(rdd):
    data = joined_rdd.collect()
    result = sorted(data, key=lambda x: (-x[1], x[0]))[:10]
    print(' '.join([city for city, avg in result]))

def sort_spark(rdd):
    result = rdd.takeOrdered(10, lambda x: (-x[1], x[0]))
    print(' '.join([city for city, avg in result]))

start_3a = time.time()
result_3a = sort_python(joined_rdd)
end_3a = time.time()

result_3b = sort_spark(joined_rdd)
end_3b = time.time()

def toCSVString(row):
    return ','.join(str(field) for field in row)

result = joined_rdd.sortBy(keyfunc= lambda x: (-x[1], x[0]), ascending= True).coalesce(1).map(toCSVString).saveAsTextFile('output_3a.csv')
os.system("echo 'city,stars' | cat - output_3a.csv/part-00000 > " + output_1)
shutil.rmtree("output_3a.csv/")

result = {}
result["m1"] = end_3a - start_3a
result["m2"] = end_3b - end_3a

with open(output_2, 'w') as fp:
    json.dump(result, fp)

if __name__ == '__main__':
    pass