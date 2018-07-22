import numpy as np
from matrix import Matrix
import math
import random

def sigmoid(x):
    return 1 / (1 + math.exp(-x))
def dsigmoid(y):
    return y * (1 - y) #sigmoid(x) * (1 - sigmoid(x))

class NeuralNetwork():
    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # set the learning rate
        self.lr = 0.1

        # inputs -> hidden layer -> outputs
        self.weights_ih = Matrix(self.hidden_nodes, self.input_nodes)
        self.weights_ho = Matrix(self.output_nodes, self.hidden_nodes)

        # biases for hidden and output layer
        self.bias_h = Matrix(self.hidden_nodes, 1)
        self.bias_o = Matrix(self.output_nodes, 1)

        # randomize weigts and biases
        self.weights_ih.randomize()
        self.weights_ho.randomize()
        self.bias_h.randomize()
        self.bias_o.randomize()

    ''' sets the learning rate '''
    def setLR(lr):
        self.lr = lr
    
    def save_best_model(self):
        # TODO: write a file with info regarding class
        pass

    ''' predicts output '''
    def feed_forward(self, input_array):
        # get the values for the hidden nodes
        inputs = Matrix.fromArray(input_array)
        hidden = Matrix.matMultiply(self.weights_ih, inputs)
        hidden.add(self.bias_h)

        # activation function on the hidden nodes
        hidden.map(sigmoid)

        # get the values for the output
        output = Matrix.matMultiply(self.weights_ho, hidden) # multiply ho wegths with hidden nodes
        output.add(self.bias_o) # add bias
        output.map(sigmoid) # sigmoid activation

        # return output as array
        return output.toArray()
    
    def predictions(self, multiple_inputs):
        # return [self.feed_forward(input) for input in multiple_inputs]
        predictions = []
        for input in mmultiple_inputs:
            predictions.append(self.feed_forward(input))
        return predictions
    
    def confusion_matrix(self, xs, ys, labels, normalize=False):
        ''' generate a confusion matrix ''' 
        ''' cols are the predictions and the rows are the vals ''' 
        cm = Matrix(len(labels), len(labels))
        
        for x, y in zip(xs, ys):
            index_prediction, index_actual = self.get_index_val(x, y)
            # print(f'{index_actual}, {index_prediction}')
            cm.data[index_actual][index_prediction] += 1
            
        if normalize:
            cm.multiply(1/len(xs))
        return cm
        
    def get_index_val(self, x, y):
        prediction = self.feed_forward(x)
        index_prediction = prediction.index(max(prediction))
        index_actual = y.index(max(y))
        return index_prediction, index_actual
    
    def compare_index_val(self, x, y):
        ''' compare indicies of individual points ''' 
        index_prediction, index_actual = self.get_index_val(x, y)
        if (index_prediction == index_actual):
            return True
        else:
            return False
            
    def get_accuracy(self, xs, ys):
        ''' compare the values of the indicies of the arrays ''' 
        score = 0        
        for x, y in zip(xs, ys):
            if self.compare_index_val(x, y):
                score += 1            
        return score / len(xs)
                
        

    ''' trains the nn '''
    def train(self, input_array, target_array):
        inputs = Matrix.fromArray(input_array)
        targets = Matrix.fromArray(target_array)

        # get values for hidden nodes
        hidden = Matrix.matMultiply(self.weights_ih, inputs)
        hidden.add(self.bias_h)
        hidden.map(sigmoid)
        # get values for output nodes
        outputs = Matrix.matMultiply(self.weights_ho, hidden)
        outputs.add(self.bias_o)
        outputs.map(sigmoid)


        # calculate the error, ERROR = TARGETS - OUTPUTS
        output_errors = Matrix.subtract(targets, outputs)

        # get derivative of output_errors: gradien = ouptuts * (1 - outputs)
        gradients = Matrix.static_map(outputs, dsigmoid)

        gradients.multiply(output_errors)
        gradients.multiply(self.lr)

        # calculate deltas: delta_w = error * hidden_t * lr
        hidden_T = Matrix.transpose(hidden)
        weights_ho_deltas = Matrix.matMultiply(gradients, hidden_T)

        # adjsut the weights
        self.weights_ho.add(weights_ho_deltas)
        # adjust bias
        self.bias_o.add(gradients)

        # hidden layer errors
        # weights hidden out tranpose
        who_t = Matrix.transpose(self.weights_ho)
        hidden_errors = Matrix.matMultiply(who_t, output_errors)

        # hidden gradient
        hidden_gradient = Matrix.static_map(hidden, dsigmoid)
        hidden_gradient.multiply(hidden_errors)
        hidden_gradient.multiply(self.lr)

        # calc input -> hidden deltas
        inputs_T = Matrix.transpose(inputs)
        weights_ih_deltas = Matrix.matMultiply(hidden_gradient, inputs_T)

        # adjust weights for input_hidden
        self.weights_ih.add(weights_ih_deltas)
        self.bias_h.add(hidden_gradient)

