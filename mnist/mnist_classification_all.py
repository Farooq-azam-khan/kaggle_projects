from process_images import get_labeled_data

import numpy as np
from sklearn import svm
from sklearn.metrics import confusion_matrix

def get_data():
    train_X, train_y    =  get_labeled_data('./data/train-images-idx3-ubyte.gz', './data/train-labels-idx1-ubyte.gz', verbose=True)  
    test_X, test_y      = get_labeled_data('./data/t10k-images-idx3-ubyte.gz', './data/t10k-labels-idx1-ubyte.gz', verbose=True)   
    return train_X, test_X, train_y, test_y 

def flatten_array(X, y):
    X_flattened = np.zeros((len(X), len(X[0])*len(X[0])), dtype=np.float32) # initalize np array)
    y_flattened = np.zeros((len(y)), dtype=np.uint8) # initalize np array)

    for indx, val in enumerate(X):
        X_flattened[indx] = np.reshape(val, 28*28)
        
    for indx, val in enumerate(y):
        train_y_flattened[indx] = y[0]
    print(train_y_flattened)
    
    return x_flattened, y_flattened

def train_svc_model(train_X, train_y):
    clf = svm.SVC(gamma=0.01)
    clf.fit(train_X, train_y)
    return clf

def confusion_matrix_svc(clf, X, y):
    y_true_svc = y
    y_pred_svc = clf.predict(X)
    cm = confusion_matrix(y_true_svc, y_pred_svc)
    print(cm)
    
def main():
    train_X, test_X, train_y, test_y = get_data()
    print('got all training and testing data')
    
    train_X_flattened = flatten_array(train_X)
    train_y_flattened = flatten_array(train_y)
    
    test_X_flattened = flatten_array(test_X)
    test_y_flattened = flatten_array(test_y)
    print('flattened data')
    
    clf = train_svc_model(train_X_flattened, train_y_flattened)
    print('fitted the model')
    
    print('confusion matrix')
    confusion_matrix_svc(clf, test_X_flattened, test_y_flattened)


if __name__ == '__main__':
    main()