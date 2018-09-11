from sklearn import model_selection, metrics
from sklearn import (svm, 
                     neighbors, 
                     ensemble, 
                    linear_model, 
                    naive_bayes, 
                    tree, 
                    discriminant_analysis)
import numpy as np
import pandas as pd
import time


from model_evaluation_function import smape

# load test_data 
from load_processed_data import get_train_test_split
train_X, test_X, train_y, test_y = get_train_test_split()



# List of MLA Algorithms
MLA = [
    # ensemble
#     ensemble.AdaBoostClassifier(),
#     ensemble.AdaBoostRegressor(),
#     ensemble.RandomForestClassifier(), 
    ensemble.RandomForestRegressor(),
#     ensemble.BaggingClassifier(),
    ensemble.GradientBoostingRegressor(),
    ensemble.ExtraTreesRegressor(),
    
    #Nearest Neighbor
#     neighbors.KNeighborsClassifier(),
    
    # svm
    svm.LinearSVR(), 
    svm.SVR(), 
    svm.NuSVR(),
    
    
    # tree
#     tree.DecisionTreeClassifier(),
    tree.DecisionTreeRegressor(),    
]


def ml_training(MLA):
    pd_dataframe = pd.DataFrame(columns=['Name', 'Train_Score', 'Test_Score', 'SAMPE_Train', 'SAMPE_Test', 'Time', 'Parameters'])
    row_number = 0
    for alg in MLA:
        alg_name = alg.__class__.__name__
        print("starting:", alg_name)

        start_time = time.time()
        alg.fit(train_X, train_y)
        end_time = time.time()
        time_taken = end_time - start_time

        train_score = alg.score(train_X, train_y)
        test_score = alg.score(test_X, test_y)
        sampe_train = smape(alg.predict(train_X), train_y)
        sampe_test = smape(alg.predict(test_X), test_y)

        # add to pandas dataframe
        pd_dataframe.loc[row_number] = [alg_name, train_score, test_score, sampe_train, sampe_test, time_taken, alg.get_params()]
        row_number+=1
        
    pd_dataframe.sort_values(by=['SAMPE_Test'], ascending=False, inplace=True)
    print('done')
    return pd_dataframe


MLA_Compare = ml_training(MLA)

print(MLA_Compare)