import pandas as pd 

from sklearn.model_selection import train_test_split

features_list = ['store', 
	                       'item', 
	                       'day', 'month', 'year', 
	                       'weekofyear', 'dayofyear', 'weekday',
	                      'weekofyear_median', 'dayofyear_median', 
	                      'store_sales_median', 'item_sales_median', 
	                      'store_item_sales_median', 
	                      'store_item_weekofyear_sales_median']

def get_test_data():

	df = pd.read_csv('./data/preprocessed_train_test_data.csv')
	test_data = df.loc[df['sales'].isna()]
	print("test.csv",test_data.shape)

	features = test_data[features_list] 

	return features

def get_features_targets():

	df = pd.read_csv('./data/preprocessed_train_test_data.csv')

	train_data = df.loc[~df['sales'].isna()]
	print("train.csv",train_data.shape)
	test_data = df.loc[df['sales'].isna()]
	print("test.csv",test_data.shape)

	features = train_data[features_list] 
	targets = train_data['sales']

	return features, targets


def get_train_test_split():
	features, targets = get_features_targets()
	train_X, test_X, train_y, test_y = train_test_split(features, targets)

	return train_X, test_X, train_y, test_y