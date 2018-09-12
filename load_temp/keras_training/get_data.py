import pandas as pd

from sklearn.model_selection import train_test_split


area_features_list = [f'area{a}' for a in range(1,6+1)]
hours_zone_id_mean = [f'h{hour}_zoneid_mean' for hour in range(1,24+1)]
hours_zone_id_median = [f'h{hour}_zoneid_median' for hour in range(1,24+1)]
hours_zone_id_sum = [f'h{hour}_zoneid_sum' for hour in range(1,24+1)]

hours_month_mean = [f'h{hour}_month_mean' for hour in range(1,24+1)]
hours_month_median = [f'h{hour}_month_median' for hour in range(1,24+1)]
hours_month_sum = [f'h{hour}_month_sum' for hour in range(1,24+1)]

hours_day_mean = [f'h{hour}_day_mean' for hour in range(1,24+1)]
hours_day_median = [f'h{hour}_day_median' for hour in range(1,24+1)]
hours_day_sum = [f'h{hour}_day_sum' for hour in range(1,24+1)]

hours_month_day_mean = [f'h{hour}_month_day_mean' for hour in range(1,24+1)]
hours_month_day_median = [f'h{hour}_month_day_median' for hour in range(1,24+1)]
hours_month_day_sum = [f'h{hour}_month_day_sum' for hour in range(1,24+1)]

hours_zoneid_month_day_mean = [f'h{hour}_zoneid_month_day_mean' for hour in range(1,24+1)]
hours_zoneid_month_day_median = [f'h{hour}_zoneid_month_day_median' for hour in range(1,24+1)]
hours_zoneid_month_day_sum = [f'h{hour}_zoneid_month_day_sum' for hour in range(1,24+1)]


features_list = ['year', 
				'month', 
				'day', 
				'weekday', 
				'week_of_year', 
				'day_of_year', 
				'is_month_start', 'is_month_end', 
				'zone_id'
] + area_features_list + hours_zone_id_mean + hours_zone_id_median + hours_zone_id_sum + hours_month_mean
features_list += hours_month_median + hours_month_sum + hours_day_mean + hours_day_median + hours_day_sum
features_list += hours_month_day_mean + hours_month_day_median + hours_month_day_sum + hours_zoneid_month_day_mean + hours_zoneid_month_day_median + hours_zoneid_month_day_sum

targets_list = [f'h{hour}' for hour in range(1,24+1)]

def get_features_targets():
	df = pd.read_csv('../data/preprocessed_train_load_data.csv')
	# print(df.columns)

	features = df[features_list]
	targets = df[targets_list]

	return features, targets


def get_train_test_split():
	features, targets = get_features_targets()
	return train_test_split(features, targets, random_state=0)

if __name__ == '__main__':
	features, targets = get_features_targets()


def get_test_data():
	pass

def get_backcasting_data():
	pass

def get_forecasting_data():
	pass

