import csv
import sys
import math
import time
import heapq
from functools import reduce
from pyspark import SparkContext
from collections import defaultdict


sc = SparkContext('local[*]', 'Task1')
sc.setLogLevel("OFF")


def get_input():
    run = 'non'
    if run == 'local':
        train = '../../dataset/yelp_train.csv'
        test = '../../dataset/yelp_test.csv'
        op = 'out.csv'
    else:
        train = sys.argv[1]
        test = sys.argv[2]
        op = sys.argv[3]
    return train, test, op


def get_rdd(path, flag=0):
    train_file = sc.textFile(path)
    header = train_file.first()
    training_rdd = train_file.filter(lambda x: x != header)
    train = training_rdd.map(lambda x:mapper(x, flag)).collect()
    return train


def mapper(x, flag):
    if flag:
        user_id, business_id = x.split(',')
        return user_id, business_id
    else:
        user_id, business_id, stars = x.split(',')
        return user_id, business_id, int(float(stars))


def users_stars_average(data):
    return data[0], reduce(lambda x, y: x+y, [x[1] for x in data[1]])/len(data[1])


def business_stars_average(data):
    return data[0], reduce(lambda x, y: x+y, [x[1] for x in data[1]])/len(data[1])


def pearson(target, user_business_map, star_map, business_user_map):
    user, business = target
    other_businesses = user_business_map[user]
    closest_businesses = []
    N = 110
    for other_business in other_businesses.difference([business]):
        common_users = business_user_map[other_business].intersection(business_user_map[business])
        if common_users:
            business_stars = [star_map[(common_user,business)] for common_user in common_users]
            other_business_stars = [star_map[(common_user, other_business)] for common_user in common_users]
            business_correlated_avg = sum(business_stars)/len(business_stars)
            other_business_correlated_avg = sum(other_business_stars)/len(other_business_stars)
            adjusted_business_stars = [star-business_correlated_avg for star in business_stars]
            adjusted_other_business_stars = [star - other_business_correlated_avg for star in other_business_stars]

            dot = sum([x*y for x, y in zip(adjusted_business_stars, adjusted_other_business_stars)])

            if sum(adjusted_business_stars) and sum(adjusted_other_business_stars):
                mag1 = math.sqrt(sum([x*x for x in adjusted_business_stars]))
                mag2 = math.sqrt(sum([x*x for x in adjusted_other_business_stars]))
                weight = dot/(mag1*mag2)
            else:
                weight = 0.19
        else:
            weight = 0.1

        if len(closest_businesses) < N:
            heapq.heappush(closest_businesses, (weight, other_business))
        else:
            heapq.heappushpop(closest_businesses, (weight, other_business))
    # Now we should have the N closest businesses
    closest_businesses = list(filter(lambda x: x[0] > 0, closest_businesses))
    # total = 0
    # for i in closest_businesses:
    #     if closest_businesses[i][0] < 0:
    #
    # predict = total/

    if closest_businesses:
        predict = reduce(lambda x, y: x+y[0]*star_map[(user, y[1])], closest_businesses, 0) \
                                    / (sum([abs(x[0]) for x in closest_businesses]))
    else:
        predict = 3.7512
    # clip
    predict = max(1, min(predict, 5))
    return (user, business), round(predict, 2)


def write_result(pred_map, out_path):
    with open(out_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["user_id", "business_id", "prediction"])
        for u,b,p in pred_map:
            writer.writerow([u, b, p])



def process(train_path, test_path, output_path):
    train = get_rdd(train_path)
    test = get_rdd(test_path, 1)
    time_rdd = time.time()
    # business, user index and reverse index
    business_id_map = {}
    id_business_map = {}
    user_id_map = {}
    id_user_map = {}

    # Assign indices
    business_index = 0
    user_index = 0

    # Useful maps
    user_business_star_map = defaultdict(set)
    user_business_map = defaultdict(set)
    business_user_star_map = defaultdict(set)  # business1: {(user1, star1), (user2, star2)}
    business_user_map = defaultdict(set)  # business: {user1, user2}
    star_map = defaultdict()

    for user, business, stars in train:
        if user not in user_id_map:
            user_id_map[user] = user_index
            id_user_map[user_index] = user
            user_index += 1
        if business not in business_id_map:
            business_id_map[business] = business_index
            id_business_map[business_index] = business
            business_index += 1
        user_id = user_id_map[user]
        business_id = business_id_map[business]
        user_business_star_map[user_id].add((business_id, stars))
        user_business_map[user_id].add(business_id)
        business_user_star_map[business_id].add((user_id, stars))
        business_user_map[business_id].add(user_id)
        star_map[(user_id, business_id)] = stars

    users_avg = sc.parallelize(user_business_star_map.items()).map(users_stars_average).collectAsMap()
    business_avg = sc.parallelize(business_user_star_map.items()).map(business_stars_average).collectAsMap()
    global_average = sum(star_map.values())/len(star_map)

    pred_dict = {}
    test_pairs = []
    all_pairs = []
    for user, business in test:
        all_pairs.append((user, business))
        if user not in user_id_map and business not in business_id_map:
            pred = global_average
        elif user not in user_id_map:
            pred = business_avg[business_id_map[business]]
        elif business not in business_id_map:
            pred = users_avg[user_id_map[user]]
        else:
            test_pairs.append((user_id_map[user], business_id_map[business]))
            continue
        pred_dict[(user, business)] = pred

    predictions = sc.parallelize(test_pairs).map(lambda x: pearson(x, user_business_map, star_map,
                                                                   business_user_map)).collectAsMap()
    result = []
    for user, business in all_pairs:
        if (user, business) in pred_dict:
            result.append((user, business, pred_dict[(user, business)]))
        else:
            result.append((user, business, predictions[(user_id_map[user], business_id_map[business])]))
    write_result(result, output_path)


if __name__ == '__main__':
    time_start = time.time()
    train_file_path, test_file_path, output_file_path = get_input()
    process(train_file_path, test_file_path, output_file_path)
    time_end = time.time()
    print("Duration {}".format(time_end-time_start))