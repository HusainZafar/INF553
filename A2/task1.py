import sys
import time
import copy
from collections import Counter
from pyspark import SparkContext
from operator import add

sc = SparkContext('local[*]', 'Task1')
#sc.setLogLevel("OFF")

def get_input():
    run = 'non'
    if run == 'local':
        case_number = 1
        support = 4
        input_file_path = '../../dataset/small2.csv'
        output_file_path = 'output.txt'
    else:
        case_number = int(sys.argv[1])
        support = int(sys.argv[2])
        input_file_path = sys.argv[3]
        output_file_path = sys.argv[4]
    return case_number, support, input_file_path, output_file_path


def test_count(file, combo):
    with open(file) as fp:
        mapping = {}
        data = fp.readline()
        data = fp.readline()
        while(data):
            k,v = map(int,data.strip().split(','))
            if k not in mapping:
                mapping[k] = set()
            mapping[k].update(set([v]))
            data = fp.readline()
        count = 0
        for k,v in mapping.items():
            flag = 1
            for item in combo:
                if item not in v:
                    flag = 0
                    break
            if flag:
               count += 1
        return(count)


def apriori(global_baskets, support):
    #print(support)
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
    #print(occurence)
    return [(k,v) for k,v in occurence.items()]

def preprocess(line, case_number):
    a,b = map(int, line.replace(' ','').split(','))
    if case_number == 1:
        return (a, set([b]))
    else:
        return (b, set([a]))

def get_result_string(result_type, data):
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

def process(case_number, support, input_file_path):
    baskets_file = sc.textFile(input_file_path)
    header = baskets_file.first()
    baskets_rdd = baskets_file.filter(lambda x: x!= header)
    baskets_rdd = baskets_rdd.map(lambda x: preprocess(x, case_number)).reduceByKey(lambda x, y: x.union(y)).cache()
    partitions = baskets_rdd.getNumPartitions()

    candidates = baskets_rdd.mapPartitions(lambda x:apriori(x, support/partitions)).reduceByKey(add).map(lambda x: x[0]).collect()
    frequent_itemsets = baskets_rdd.mapPartitions(lambda x: check_frequent(x, candidates)).reduceByKey(add)\
        .filter(lambda x: x[1]>=support).map(lambda x: x[0]).collect()
    candidates_string = get_result_string("Candidates:\n", candidates)
    frequent_itemsets_string = get_result_string("Frequent Itemsets:\n", frequent_itemsets)

    #print(len(candidates))
    #print(len(frequent_itemsets))
 
    with open(output_file_path, 'w') as fp:
        fp.writelines(candidates_string + ["\n\n"] + frequent_itemsets_string)


if __name__ == '__main__':
    time_start = time.time()
    case_number, support, input_file_path, output_file_path = get_input()
    process(case_number, support, input_file_path)
    time_end = time.time()
    print("Duration {}".format(time_end-time_start))