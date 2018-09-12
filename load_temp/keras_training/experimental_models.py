from get_data import get_train_test_split


train_X, test_X, train_y, test_y = get_train_test_split()
print('loaded train/test data')

EPOCHS = 35
BATCH_SIZE = 32

from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import SGD, RMSprop, Adadelta, Adam
from keras.callbacks import TensorBoard, ReduceLROnPlateau


# callbacks 
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=2, min_lr=0.001)

model = Sequential()
# add input
model.add(Dense(128, input_dim=len(train_X.columns)))

# hidden layers
model.add(Dense(128, activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='sigmoid'))
model.add(Dense(128, activation='sigmoid'))
model.add(Dense(128, activation='sigmoid'))
model.add(Dense(128, activation='sigmoid'))
model.add(Dense(128, activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='sigmoid'))
model.add(Dense(128, activation='sigmoid'))
model.add(Dense(128, activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='sigmoid'))
model.add(Dense(128, activation='sigmoid'))
model.add(Dense(128, activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='sigmoid'))
model.add(Dense(128, activation='sigmoid'))
model.add(Dense(128, activation='sigmoid'))
model.add(Dropout(0.2))

# output
model.add(Dense(train_y.shape[1], activation='relu'))

# compiling model 
model.compile(loss='mse', 
			optimizer=Adadelta(lr=0.03))

print('fitting model')
model.fit(train_X.values, train_y.values, 
			batch_size=BATCH_SIZE, 
			epochs=EPOCHS, 
			shuffle=True,
			validation_data=[test_X, test_y], 
			callbacks=[reduce_lr]
		)

MODEL_SCORE = model.evaluate(test_X, test_y)
MODEL_NAME = f'sigmoid_10_nodes_128_dropout_20_score_{MODEL_SCORE}.model'
model.save(f'./models/{MODEL_NAME}')