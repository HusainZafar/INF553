import sys
import time
from collections import Counter
from pyspark import SparkContext
from operator import add
import shutil

sc = SparkContext('local[8]', 'Task1')
#sc.setLogLevel("OFF")

def get_input():
    run = 'non'
    if run == 'local':
        filter_threshold = 20
        support = 50
        input_file_path = '../../dataset/ta_feng_all_months_merged.csv'
        output_file_path = 'output-task2.txt'
    else:
        filter_threshold = int(sys.argv[1])
        support = int(sys.argv[2])
        input_file_path = sys.argv[3]
        output_file_path = sys.argv[4]
    return filter_threshold, support, input_file_path, output_file_path


def preprocess_raw(x):
    fields = x.split(",")
    date = fields[0].strip('" ')
    if date == "TRANSACTION_DT":
        return "header"
    date = date[:-4]+date[-2:]
    cust_id = fields[1].strip('" ')
    product_id = fields[5].strip('" ')
    return (date+"-"+cust_id, set([int(product_id)]))


def preprocess_intermediate(x):
    fields = x.split(",")
    transaction = fields[0].strip('" ')
    product = fields[1].strip('" ')
    return (transaction, set([int(product)]))

def apriori(global_baskets, support):
    baskets = []
    global_occurence = Counter()
    occurence = Counter()
    result = []
    size = 1

    #baskets = [values for key, values in global_baskets]
    for key, values in global_baskets:
        baskets.append(values)
        occurence.update(iter(values))
        global_occurence.update(occurence)
    current = set([k for k in occurence if occurence[k] >= support])
    result += [(k,) for k in current]
    #print(occurence)

    while current:
        size += 1
        occurence = Counter()
        for b in range(len(baskets)):
            elements = list(baskets[b] - (baskets[b] - current))
            current_basket = set()
            temp = {}
            for i in range(len(elements)):
                for j in range(i+1, len(elements)):
                    if size == 2:
                        union = set([elements[i]]).union(set([elements[j]]))
                    else:
                        union = set(elements[i]).union(set(elements[j]))
                    if len(union) == size:
                        tuple_sorted = tuple(sorted(union, key=lambda x: str(x)))
                        current_basket.add(tuple_sorted)
                        temp[tuple_sorted] = 1
            occurence += temp
            baskets[b] = current_basket.copy()

        frequent = [k for k in occurence if occurence[k] >= support]
        #print(occurence)
        global_occurence.update(occurence)
        current = set(frequent)
        result += frequent
    #print(global_occurence)
    ret = [(pair,1) for pair in result]
    return(ret)

def check_frequent(global_baskets, candidates):
    baskets = list(global_baskets)
    #print(baskets)

    occurence = {}
    for candidate in candidates:
        for basket_key, basket_values in baskets:
            if set(candidate) <= basket_values:
                occurence[candidate] = occurence.get(candidate, 0) + 1
    return [(k,v) for k,v in occurence.items()]


def get_result_string(result_type, data):
    # data_sorted = copy.deepcopy(data)
    # for idx in range(len(data_sorted)):
    #     if len(data_sorted[idx]) == 1:
    #         continue
    #     data_sorted[idx] = tuple(sorted(data_sorted[idx], key = lambda x: str(x)))
    data_sorted = sorted(data, key=lambda x: (len(x), str(x)))

    result = [result_type]
    if not data_sorted:
        return result
    max_size = len(data_sorted[-1])
    size = 1
    result_string = ""
    for candidate in data_sorted:
        if len(candidate) == size:
            if len(candidate) == 1:
                result_string += "('" + str(candidate[0]) + "')" + ","
            else:
                result_string += str(tuple(map(str, candidate))) + ","
        if len(candidate) > size:
            result.append(result_string[:-1]+"\n\n")
            result_string = str(tuple(map(str, candidate))) + ","
            size += 1
    if result_string[-1] == ',':
        result.append(result_string[:-1])
    else:
        result.append(result_string)
    return result


def process(filter_threshold, support, input_file_path, output_file_path):
    raw_file = sc.textFile(input_file_path)
    raw_rdd = raw_file.map(lambda x: preprocess_raw(x)).filter(lambda x: x != "header")\
        .reduceByKey(lambda x, y: x.union(y))

    filtered_rdd = raw_rdd.filter(lambda x: len(x[1]) > filter_threshold)
    partitions = filtered_rdd.getNumPartitions()

    candidates = filtered_rdd.mapPartitions(lambda x: apriori(x, support / partitions)).reduceByKey(add).map(
        lambda x: x[0]).collect()
    frequent_itemsets = filtered_rdd.mapPartitions(lambda x: check_frequent(x, candidates)).reduceByKey(add) \
        .filter(lambda x: x[1] >= support).map(lambda x: x[0]).collect()
    candidates_string = get_result_string("Candidates:\n", candidates)
    frequent_itemsets_string = get_result_string("Frequent Itemsets:\n", frequent_itemsets)

    with open(output_file_path, 'w') as fp:
        fp.writelines(candidates_string + ["\n\n"] + frequent_itemsets_string)


if __name__ == '__main__':
    time_start = time.time()
    filter_threshold, support, input_file_path, output_file_path = get_input()
    process(filter_threshold, support, input_file_path, output_file_path)
    time_end = time.time()
    print("Duration {}".format(time_end - time_start))