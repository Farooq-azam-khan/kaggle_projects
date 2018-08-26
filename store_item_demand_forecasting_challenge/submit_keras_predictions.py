import pandas as pd

# load best model
from keras.models import load_model
from model_evaluation_function import smape

MODEL_NAME = 'BEST_MODEL.model'
model = load_model(f'./models/{MODEL_NAME}', 
					custom_objects={'smape': smape})
# print(model.summary())



# load test_data 
from load_processed_data import get_test_data
test_data_featues = get_test_data()

# create predictions
predictions = model.predict(test_data_featues)
print(predictions[:10])

# submission
submission = pd.read_csv('./data/sample_submission.csv.zip')
submission['sales'] = predictions
submission.to_csv('./data/submission.csv', index=False)