from process_images import get_labeled_data

import numpy as np
from sklearn import svm
from sklearn.metrics import confusion_matrix

def main():
    train_X, train_y    =  get_labeled_data('./data/train-images-idx3-ubyte.gz', './data/train-labels-idx1-ubyte.gz', verbose=True)  
    test_X, test_y      = get_labeled_data('./data/t10k-images-idx3-ubyte.gz', './data/t10k-labels-idx1-ubyte.gz', verbose=True)   
    
    # flatten arrays
    # flatten 3d array to 2d array for machine learning
    # was [image][28][28] now it will be [image][28*28] because models does not care about image dimentionality 
    train_X_flattened = np.zeros((len(train_X), len(train_X[0])*len(train_X[0])), dtype=np.float32) # initalize np array)
    train_y_flattened = np.zeros((len(train_y)), dtype=np.uint8) # initalize np array)

    for indx, X in enumerate(train_X):
        train_X_flattened[indx] = np.reshape(X, 28*28)
        
    for indx, y in enumerate(train_y):
        train_y_flattened[indx] = y[0]
    print(train_y_flattened)
    
    # now for testing data
    test_X_flattened = np.zeros((len(test_X), len(test_X[0])*len(test_X[0])), dtype=np.float32) # initalize np array)
    test_y_flattened = np.zeros((len(test_y)), dtype=np.uint8) # initalize np array)

    for indx, X in enumerate(test_X):
        test_X_flattened[indx] = np.reshape(X, 28*28)
        
    for indx, y in enumerate(test_y):
        test_y_flattened[indx] = y[0]

    print(test_X_flattened)
    print(test_y_flattened)
    
    print('got all training and testing data')
    clf = svm.SVC(gamma=0.001)
    clf.fit(train_X_flattened, train_y_flattened)
    print('fitted the model')
    
    y_true_svc = test_y_flattened
    y_pred_svc = clf.predict(test_X_flattened)
    cm = confusion_matrix(y_true_svc, y_pred_svc)
    print(clf.score(test_X_flattened, test_y_flattened))
    print(cm)

if __name__ == '__main__':
    main()