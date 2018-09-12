from get_data import get_train_test_split

train_X, test_X, train_y, test_y = get_train_test_split()
print('data loaded')

from sklearn import model_selection, metrics
from sklearn import (svm, 
                     neighbors, 
                     ensemble, 
                     multioutput,
                    linear_model, 
                    naive_bayes, 
                    tree, 
                    discriminant_analysis)


# List of MLA Algorithms
MLA = [
#     ensemble
    ensemble.AdaBoostRegressor(),
    ensemble.RandomForestRegressor(),
    ensemble.GradientBoostingRegressor(),
    ensemble.ExtraTreesRegressor(),
    
#     Nearest Neighbor
    
#     svm
    svm.LinearSVR(), 
#     svm.SVR(), 
#     svm.NuSVR(),
    
    
    # tree
    tree.DecisionTreeRegressor(),    
]


for alg in MLA: 
    alg_name = alg.__class__.__name__
    print("starting:", alg_name)
    clf = multioutput.MultiOutputRegressor(alg)
    clf.fit(train_X, train_y)
    
    print("\ttrainig score:", clf.score(train_X, train_y))
    print("\ttesting score:",clf.score(test_X, test_y))
    print("\ttrainig mse:", metrics.mean_squared_error(y_true=train_y, y_pred=clf.predict(train_X)))
    print("\ttesting mse:",metrics.mean_squared_error(y_true=test_y, y_pred=clf.predict(test_X)))
    print('-'*50)
