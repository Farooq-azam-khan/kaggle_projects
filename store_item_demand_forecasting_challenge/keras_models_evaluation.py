
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import SGD, RMSprop, Adadelta, Adam
from keras.callbacks import TensorBoard, ReduceLROnPlateau

# 
from model_evaluation_function import smape
from load_processed_data import get_train_test_split

print('loading train/test data')
train_X, test_X, train_y, test_y = get_train_test_split()

EPOCHS = 10
BATCH_SIZE = 32


optimizers = [SGD, RMSprop, Adadelta, Adam]
learning_rate = [0.1, 0.03, 0.001]
nodes_per_dense_layer = [32, 64, 128]
dense_layers = [1, 2, 3, 4]
activation_functions = ['tanh', 'elu', 'relu', 'linear']
dropout_rates = [0.1, 0.25]

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=2, min_lr=0.001)

print('starting model creation')
for nodes in nodes_per_dense_layer:
	for dense_layer in dense_layers:
		for dropout_rate in dropout_rates:

			# init model 
			model = Sequential()
			# add input
			model.add(Dense(nodes, input_dim=len(train_X.columns)))

			# add dense layers
			for dk in range(dense_layer):
				model.add(Dense(nodes, activation='relu'))
				model.add(Dropout(dropout_rate))

			
				
			# output
			model.add(Dense(1, activation='relu'))

			print('compiling model')
			model.compile(optimizer=Adadelta(lr=0.03), 
						loss=smape
						)

			

			print('initalizing TensorBoard')
			TENSORBOARD_MODEL = f'Sotre_Item_{nodes}-nodes_{dense_layer}-denseL_{dropout_rate}-dropout'
			tensorboard = TensorBoard(log_dir=f'./logs/{TENSORBOARD_MODEL}')

			print('fitting model')
			model.fit(train_X.values, train_y.values, 
			          batch_size=BATCH_SIZE, 
			          epochs=EPOCHS, 
			          shuffle=True,
			          validation_data=[test_X, test_y], 
			          callbacks=[tensorboard, reduce_lr]
			         )

			# Save model
			model_score = model.evaluate(test_X, test_y)
			MODEL_NAME = f'{TENSORBOARD_MODEL}_{model_score}.model'
			model.save(f'./models/{MODEL_NAME}')
