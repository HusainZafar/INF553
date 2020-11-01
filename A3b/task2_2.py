import sys
import json
import time
import pandas as pd
from sklearn import preprocessing
import xgboost as xgb
from pyspark import SparkContext

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


def process(train_set, test_set, output_path):
	clean_df = clean(train_set.copy())
	clean_test = clean(test_set.copy())

	train_y = clean_df.review_score.values
	train_x = clean_df.drop(["review_score","user_id","business_id"], axis=1, errors='ignore')
	train_x = train_x.values
	model = xgb.XGBRegressor()
	model.fit(train_x, train_y)

	test = clean_test.drop(['stars',"user_id","business_id"], axis=1, errors='ignore')
	test_x = test.values
	output = model.predict(data=test_x)
	final_df = pd.DataFrame()

	final_df["user_id"] = test_set.user_id.values
	final_df["business_id"] = test_set.business_id.values
	final_df["prediction"] = output
	final_df["prediction"] = final_df["prediction"].clip(1,5)
	#final_df["prediction"] = final_df["prediction"].round(2)
	final_df.to_csv(output_path, index=False)


if __name__ == '__main__':
	time_start = time.time()
	#path = '../../dataset/'
	path = sys.argv[1]
	user_rdd = get_user_features(path + 'user.json').collect()
	businesss_rdd = get_business_features(path + 'business.json').collect()
	review_rdd = get_review_features(path + 'review_train.json').collect()

	user_df = pd.DataFrame(user_rdd, columns=['user_id', 'user_total_reviews', 'user_total_useful_review',
											  'user_fans', 'user_average_stars'])
	businesss_df = pd.DataFrame(businesss_rdd, columns=['business_id',
											'business_average_stars', 'business_total_reviews'])
	train_df = pd.DataFrame(review_rdd, columns=['business_id', 'user_id', 'review_score'])
	train_df = train_df.merge(user_df,  on='user_id')
	train_df = train_df.merge(businesss_df, on='business_id')

	#test_df = pd.read_csv(path + 'yelp_val.csv')
	test_df = pd.read_csv(sys.argv[2])
	test_df = test_df.merge(user_df, on='user_id')
	test_df = test_df.merge(businesss_df, on='business_id')
	test_df.fillna(0, inplace=True)
	#process(train_df, test_df, 'out.csv')
	process(train_df, test_df, sys.argv[3])

	time_end = time.time()
	print("Duration {}".format(time_end-time_start))