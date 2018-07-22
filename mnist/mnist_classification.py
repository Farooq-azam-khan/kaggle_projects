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
            


def get_accuracy(xs, ys):
    score = 0 
    for x, y in zip(xs, ys):
        
        # prediction number
        prediction_arr = nn.feed_forward(x)
        max_val_pred = max(prediction_arr)
        index_max_val = prediction_arr.index(max_val_pred)
        
        # actual number
        max_val_y = max(y)
        actual_y = y.index(max_val_y)
        
        if actual_y == index_max_val:
            score += 1
        
    return score / len(xs)

def get_digits(ys):
    dig_ys = []
    for y in ys:
        dig_ys.append(np.argmax(y))
    return dig_ys

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix for Digits',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    
def main():  
    nn_training()
    
    # getting confusion matrix
    labels = [0,1,2,3,4,5,6,7,8,9]
    ys_true = get_digits(target)
    
    nn_predictions = []
    for x in data:
        nn_predictions.append(nn.feed_forward(x))
        
    ys_pred = get_digits(nn_predictions)
    
    cm = confusion_matrix(y_true=ys_true, y_pred=ys_pred, labels=labels)
    print(cm)
    print(get_accuracy(data, get_one_hot_array(target)))
    
    plt.figure()    
    plot_confusion_matrix(cm, classes=labels, normalize=False)

    plt.show() 

    
    
def nn_training():
    for epoch in range(10):
        train_X, test_X, train_y, test_y = get_split()
        for x, y in zip(train_X, train_y):
            nn.train(x, y)
        training_accuracy = get_accuracy(train_X, train_y)
        testing_accuracy = get_accuracy(test_X, test_y)
        print(f"Epoch: {epoch+1} ---> train_acc: {training_accuracy:.2} ---> test_acc: {testing_accuracy:.2}")
        
    
if __name__ == '__main__':
    main()
