from keras.models import load_model

MODEL_NAME = 'sigmoid_10_nodes_128_dropout_0.2.model'
model = load_model(f'./models/{MODEL_NAME}')


from get_data import get_train_test_split
train_X, test_X, train_y, test_y = get_train_test_split()
print("loaded testing data")

print('predicting on: ')
print(train_X.head(1))
print('prediction: ')
print(model.predict([train_X.head(1)]))
print('actual value: ')
print(train_y.head(1))


