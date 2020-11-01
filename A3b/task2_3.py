import csv
import sys
import math
import time
import heapq
import json
import pandas as pd
import xgboost as xgb
from functools import reduce
from pyspark import SparkContext
from sklearn import preprocessing
from collections import defaultdict

sc = SparkContext('local[*]', 'task2_2')
pd.set_option('display.max_columns', None)
sc.setLogLevel("OFF")


def get_user_features(path):
    def mapper(row):
        record = json.loads(row)
        user_id = record['user_id']
        user_total_reviews = record['review_count']
        user_total_useful_review = record['useful']
        user_fans = record['fans']
        user_average_stars = record['average_stars']
        return user_id, user_total_reviews, user_total_useful_review, user_fans, user_average_stars

    user_rdd = sc.textFile(path).map(mapper)
    return user_rdd


def get_business_features(path):
    def mapper(row):
        record = json.loads(row)
        business_id = record['business_id']
        business_average_stars = record['stars']
        business_total_reviews = record['review_count']
        return business_id, business_average_stars, business_total_reviews

    business_rdd = sc.textFile(path).map(mapper)
    return business_rdd


def get_review_features(path):
    def mapper(row):
        record = json.loads(row)
        business_id = record['business_id']
        user_id = record['user_id']
        review_score = record['stars']
        return business_id, user_id, review_score

    review_rdd = sc.textFile(path).map(mapper)
    return review_rdd


def clean(data):
    for f in data.columns:
        if data[f].dtype == 'object':
            label = preprocessing.LabelEncoder()
            label.fit(list(data[f].values))
            data[f] = label.transform(list(data[f].values))
    return data


def get_rdd(path, flag=0):
    train_file = sc.textFile(path)
    header = train_file.first()
    training_rdd = train_file.filter(lambda x: x != header)
    train = training_rdd.map(lambda x: mapper(x, flag)).collect()
    return train


def mapper(x, flag):
    if flag:
        user_id, business_id = x.split(',')
        return user_id, business_id
    else:
        user_id, business_id, stars = x.split(',')
        return user_id, business_id, int(float(stars))


def users_stars_average(data):
    return data[0], reduce(lambda x, y: x + y, [x[1] for x in data[1]]) / len(data[1])


def business_stars_average(data):
    return data[0], reduce(lambda x, y: x + y, [x[1] for x in data[1]]) / len(data[1])


def pearson(target, user_business_map, star_map, business_user_map):
    user, business = target
    other_businesses = user_business_map[user]
    closest_businesses = []
    N = 110
    for other_business in other_businesses.difference([business]):
        common_users = business_user_map[other_business].intersection(business_user_map[business])
        if common_users:
            business_stars = [star_map[(common_user, business)] for common_user in common_users]
            other_business_stars = [star_map[(common_user, other_business)] for common_user in common_users]
            business_correlated_avg = sum(business_stars) / len(business_stars)
            other_business_correlated_avg = sum(other_business_stars) / len(other_business_stars)
            adjusted_business_stars = [star - business_correlated_avg for star in business_stars]
            adjusted_other_business_stars = [star - other_business_correlated_avg for star in other_business_stars]

            dot = sum([x * y for x, y in zip(adjusted_business_stars, adjusted_other_business_stars)])

            if sum(adjusted_business_stars) and sum(adjusted_other_business_stars):
                mag1 = math.sqrt(sum([x * x for x in adjusted_business_stars]))
                mag2 = math.sqrt(sum([x * x for x in adjusted_other_business_stars]))
                weight = dot / (mag1 * mag2)
            else:
                weight = 0.19
        else:
            weight = 0.1

        if len(closest_businesses) < N:
            heapq.heappush(closest_businesses, (weight, other_business))
        else:
            heapq.heappushpop(closest_businesses, (weight, other_business))
    closest_businesses = list(filter(lambda x: x[0] > 0, closest_businesses))

    if closest_businesses:
        predict = reduce(lambda x, y: x + y[0] * star_map[(user, y[1])], closest_businesses, 0) \
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
        for u, b, p in pred_map:
            writer.writerow([u, b, p])


def process_item_based(train_path, test_path):
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
    global_average = sum(star_map.values()) / len(star_map)

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
            result.append(pred_dict[(user, business)])
        else:
            result.append(predictions[(user_id_map[user], business_id_map[business])])
    item_based_df = pd.DataFrame(result, columns=["item_preds"])
    # Just score column
    return item_based_df


def process_model(train_set, test_set):
    clean_df = clean(train_set.copy())
    clean_test = clean(test_set.copy())

    train_y = clean_df.review_score.values
    train_x = clean_df.drop(["review_score", "user_id", "business_id"], axis=1, errors='ignore')
    train_x = train_x.values
    model = xgb.XGBRegressor()
    model.fit(train_x, train_y)

    # clean_test.fillna((-999), inplace=True)
    test = clean_test.drop(['stars', "user_id", "business_id"], axis=1, errors='ignore')
    test_x = test.values
    output = model.predict(data=test_x)
    final_df = pd.DataFrame()

    final_df["user_id"] = test_set.user_id.values
    final_df["business_id"] = test_set.business_id.values
    final_df["model_preds"] = output
    final_df["model_preds"] = final_df["model_preds"].clip(1, 5)

    return final_df


if __name__ == '__main__':
    time_start = time.time()
    path = sys.argv[1]
    user_rdd = get_user_features(path + 'user.json').collect()
    businesss_rdd = get_business_features(path + 'business.json').collect()
    review_rdd = get_review_features(path + 'review_train.json').collect()

    user_df = pd.DataFrame(user_rdd, columns=['user_id', 'user_total_reviews', 'user_total_useful_review',
                                              'user_fans', 'user_average_stars'])
    businesss_df = pd.DataFrame(businesss_rdd, columns=['business_id',
                                                        'business_average_stars', 'business_total_reviews'])
    train_df = pd.DataFrame(review_rdd, columns=['business_id', 'user_id', 'review_score'])
    train_df = train_df.merge(user_df, on='user_id')
    train_df = train_df.merge(businesss_df, on='business_id')

    test_df = pd.read_csv(sys.argv[2])
    test_df = test_df.merge(user_df, on='user_id')
    test_df = test_df.merge(businesss_df, on='business_id')
    test_df.fillna(0, inplace=True)

    model_df = process_model(train_df, test_df)
    item_based_df = process_item_based(sys.argv[1]+'yelp_train.csv', sys.argv[2])

    model_df['item_preds'] = item_based_df
    alpha = 0.01
    model_df['prediction'] = alpha*item_based_df['item_preds'] + (1-alpha)*model_df['model_preds']

    model_df[["user_id", "business_id", "prediction"]].to_csv(sys.argv[3], index=False)
    time_end = time.time()
    print("Duration {}".format(time_end - time_start))
