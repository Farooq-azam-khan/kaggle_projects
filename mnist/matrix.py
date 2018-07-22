import numpy as np
import random
import math

class Matrix():
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols

        self.data = []
        for i in range(self.rows):
            self.data.append([])
            for j in range(self.cols):
                self.data[i].append(0)

    ''' returns a matrix object with its values '''
    def copy(self):
        result = Matrix(self.rows, self.cols)
        for i in range(result.rows):
            for j in range(result.cols):
                result.data[i][j] = self.data[i][j]

    ''' converts an array into a matrix object '''
    @staticmethod
    def fromArray(arr):
        result = Matrix(len(arr), 1)
        for i in range(result.rows):
            result.data[i][0] = arr[i]
        return result

    ''' subtracts matricies elementwise '''
    @staticmethod
    def subtract(a, b):
        if (a.rows != b.rows or a.cols != b.cols):
            print("rows and cols must match")
            return None
        else:
            result = Matrix(a.rows, a.cols)
            for i in range(result.rows):
                for j in range(result.cols):
                    result.data[i][j] = a.data[i][j] - b.data[i][j]
            return result

    ''' converts a matrix to an array '''
    def toArray(self):
        arr = []
        for i in range(self.rows):
            for j in range(self.cols):
                arr.append(self.data[i][j])
        return arr

    ''' create random values for the matrix '''
    def randomize(self):
        for i in range(0,self.rows):
            for j in range(0,self.cols):
                self.data[i][j] = random.uniform(-1, 1)
        return self

    ''' adds matrix to this matrix elementwise or scalar wise '''
    def add(self, n):
        # element wise addition
        if (type(n) == Matrix):
            if self.rows != n.rows or self.cols != n.cols:
                print("rows and cols must match")
                return None
            else:
                for i in range(self.rows):
                    for j in range(self.cols):
                        self.data[i][j] += n.data[i][j]
                return self
        else:
            for i in range(self.rows):
                for j in range(self.cols):
                    self.data[i][j] += n
            return self


    ''' transposes a matrix '''
    @staticmethod
    def transpose(matrix):
        result = Matrix(matrix.cols, matrix.rows)
        for i in range(matrix.rows):
            for j in range(matrix.cols):
                result.data[j][i] = matrix.data[i][j]
        return result

    ''' multiplies two matrices through a dot product '''
    @staticmethod
    def matMultiply(a,b):
        if a.cols != b.rows:
            print ("cols of first matrix must match rows of second matrix")
            return None
        else:
            result = Matrix(a.rows, b.cols)
            for i in range(a.rows):
                for j in range(b.cols):
                    sum = 0
                    for k in range(a.cols):
                        sum += a.data[i][k] * b.data[k][j]
                    result.data[i][j] = sum
            return result

    ''' multiplies activationfunct(x) * matrix '''
    def map(self, func):
        for i in range(self.rows):
            for j in range(self.cols):
                val = self.data[i][j]
                self.data[i][j] = func(val)
        return self

    ''' map: calculate f(index_i_j) '''
    @staticmethod
    def static_map(matrix, func):
        result = Matrix(matrix.rows, matrix.cols)
        for i in range(matrix.rows):
            for j in range(matrix.cols):
                val = matrix.data[i][j]
                result.data[i][j] = func(val)
        return result


    ''' multiplies matrix to this matrix with the hadamard product and scalar product '''
    def multiply(self, n):
        if (type(n) == Matrix):
            # hadamard product
            if self.rows != n.rows or self.cols != n.cols:
                print("rows and cols must match")
                return None
            else:
                for i in range(self.rows):
                    for j in range(self.cols):
                        self.data[i][j] *= n.data[i][j]
                return self
        else:
            # scalar product
            for i in range(self.rows):
                for j in range(self.cols):
                    self.data[i][j] *= n
            return self

    ''' to string method '''
    def __repr__(self):
        result = ""
        for i in range(0,self.rows):
            result += "["
            for j in range(0,self.cols):
                if j == self.cols-1:
                    result += "{:3.2f}".format(self.data[i][j])
                else:
                    result += "{:3.2f}, ".format(self.data[i][j])
            result += "]\n"
        return result




def main():
    m1 = Matrix(2,2)
    m2 = Matrix(2,2)

    print(m1)
    m1.activationFunction("sigmoid")
    print(m1)
    # print(m1)
    # print(m2)
    # print(Matrix.matMultiply(m1, m2))


if __name__ == "__main__":
    main()
