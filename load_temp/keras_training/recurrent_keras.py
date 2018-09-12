

from get_data import get_train_test_split


train_X, test_X, train_y, test_y = get_train_test_split()
print('loaded train/test data')

import numpy as np
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.values.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.values.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

EPOCHS = 35
BATCH_SIZE = 32

from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.optimizers import SGD, RMSprop, Adadelta, Adam
from keras.callbacks import TensorBoard, ReduceLROnPlateau


# callbacks 
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=2, min_lr=0.001)

nodes = 128
activation = 'relu'
dropout_rate = 0.2 

model = Sequential()
# add input
model.add(LSTM(100, 
	input_shape=(train_X.shape[1], train_X.shape[2]), ))

# hidden layers
model.add(Dense(nodes, activation=activation))
model.add(Dropout(dropout_rate))
model.add(Dense(nodes, activation=activation))
model.add(Dropout(dropout_rate))
model.add(Dense(nodes, activation=activation))
model.add(Dense(nodes, activation=activation))
model.add(Dense(nodes, activation=activation))
model.add(Dense(nodes, activation=activation))
model.add(Dense(nodes, activation=activation))
model.add(Dropout(dropout_rate))
model.add(Dense(nodes, activation='relu'))
model.add(Dropout(dropout_rate))


# output
model.add(Dense(train_y.shape[1], activation='linear'))

# compiling model 
model.compile(loss="mse", 
			optimizer = Adam(lr=0.001, decay=1e-6),)

print('fitting model')
model.fit(train_X, train_y, 
			batch_size=BATCH_SIZE, 
			epochs=EPOCHS, 
			shuffle=True,
			validation_data=[test_X, test_y], 
			callbacks=[reduce_lr]
		)

MODEL_SCORE = model.evaluate(test_X, test_y)
MODEL_NAME = f'recurrent_{MODEL_SCORE}.model'
model.save(f'./models/{MODEL_NAME}')