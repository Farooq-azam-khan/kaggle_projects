import numpy as np
from sklearn import datasets
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn import svm

import itertools

# Visualization
import matplotlib.pyplot as plt

from neural_network import NeuralNetwork

mnist = datasets.load_digits()
data = mnist.data
target = mnist.target

# TODO: use tensorflow
# TODO: get more data
# TODO: visualize
# TODO: get confusion matrix
nn = NeuralNetwork(64, 30, 10)


def get_split():
    train_X, test_X, train_y, test_y = model_selection.train_test_split(data, target)
    one_hot_train_y = get_one_hot_array(train_y)
    one_hot_test_y = get_one_hot_array(test_y)
    # print(train_y[0], one_hot_train_y[0])
    return train_X, test_X, one_hot_train_y, one_hot_test_y

def get_one_hot_array(arr):
    ret_arr = []
    # make one hot array for train_y and test_y
    for y in arr:
        add_arr = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        if y == 1:
            add_arr = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        elif y == 2:
            add_arr = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        elif y == 3:
            add_arr = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        elif y == 4:
            add_arr = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
        elif y == 5:
            add_arr = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
        elif y == 6:
            add_arr = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
        elif y == 7:
            add_arr = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
        elif y == 8:
            add_arr = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
        elif y == 9:
            add_arr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
            
        ret_arr.append(add_arr)
            
    return ret_arr
            

def get_digits(ys):
    dig_ys = []
    for y in ys:
        dig_ys.append(np.argmax(y))
    return dig_ys
    
    
def main():  
    nn_training()
    see_cm()
    svm_Classification()
    
    
def svm_Classification():
    train_X, test_X, train_y, test_y = model_selection.train_test_split(data, target, random_state=0)
    clf = svm.SVC(gamma=0.001)
    clf.fit(train_X, train_y)
    y_true_svc = test_y
    y_pred_svc = clf.predict(test_X)
    cm = confusion_matrix(y_true_svc, y_pred_svc)
    print(clf.score(test_X, test_y))
    print(cm)
    

def see_cm():
    # getting confusion matrix
    train_X, test_X, train_y, test_y = get_split()
    labels = [0,1,2,3,4,5,6,7,8,9]
    cm = nn.confusion_matrix(train_X, train_y, labels, normalize=True)
    print(cm)
    
def nn_training():
    max_accuracy = 0.0
    print("started training")
    for epoch in range(15):
        train_X, test_X, train_y, test_y = get_split()
        for x, y in zip(train_X, train_y):
            nn.train(x, y)
        training_accuracy = get_accuracy(train_X, train_y)
        testing_accuracy = get_accuracy(test_X, test_y)
        if testing_accuracy > max_accuracy:
            max_accuracy = testing_accuracy
        if epoch%5==0:
            print(f"Epoch: {epoch+1} ---> train_acc: {training_accuracy:.2} ---> test_acc: {testing_accuracy:.2}")
    print(f"finished training with max acc: {max_accuracy}")
        
    
if __name__ == '__main__':
    main()
