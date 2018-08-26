
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import SGD, RMSprop, Adadelta, Adam
from keras.callbacks import TensorBoard, ReduceLROnPlateau

# 
from model_evaluation_function import smape
from load_processed_data import get_train_test_split

print('loading train/test data')
train_X, test_X, train_y, test_y = get_train_test_split()

EPOCHS = 20
BATCH_SIZE = 32


reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=2, min_lr=0.001)

print('starting model creation')

# init model 
model = Sequential()
# add input
model.add(Dense(32, input_dim=len(train_X.columns)))

# add dense layers
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.1))


	
# output
model.add(Dense(1, activation='relu'))

print('compiling model')
model.compile(optimizer=Adadelta(lr=0.1), 
			loss=smape
			)


print('fitting model')
model.fit(train_X.values, train_y.values, 
          batch_size=BATCH_SIZE, 
          epochs=EPOCHS, 
          shuffle=True,
          validation_data=[test_X, test_y], 
          callbacks=[reduce_lr]
         )

# save model
model_score = model.evaluate(test_X, test_y)
print(f'Score: {model_score} \n {model_score*200}')
MODEL_NAME = f'BEST_MODEL.model'
model.save(f'./models/{MODEL_NAME}')