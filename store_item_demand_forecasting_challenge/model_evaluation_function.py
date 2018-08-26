import keras.backend as K


def custom_smape(x, x_):
    return K.mean(2*K.abs(x-x_)/(K.abs(x)+K.abs(x_)))*100

def smape(a, f):
	numerator = K.abs(f - a) 
	denominator = (K.abs(a) + K.abs(f)) / 2
	if denominator == 0:
		denominator = 0.000000001
	val = K.mean(numerator / denominator)
	return val