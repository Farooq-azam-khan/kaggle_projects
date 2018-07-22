import gzip
from struct import unpack
import numpy as np

def get_labeled_data(imagefile, labelfile, verbose=False):
    '''READ INPUT-VECTOR (imagefile) and target class(label, 0-9)'''
    
    images = gzip.open(imagefile, 'rb')
    labels = gzip.open(labelfile, 'rb')
    
    # print(images)
    # print(labels)
    
    # get metadata for images 
    images.read(4) # skip the magic_number
    number_of_images = images.read(4)
    number_of_images = unpack('>I', number_of_images)[0]
    
    rows = images.read(4)
    rows = unpack('>I', rows)[0]
    
    cols = images.read(4)
    cols = unpack('>I', cols)[0]
    
    
    # get metadata for labels
    labels.read(4) # skip magic_number
    number_of_labels = labels.read(4)
    number_of_labels = unpack('>I', number_of_labels)[0]
    
    if verbose: 
        print(f"number of images: {number_of_images}, number of labels: {number_of_labels}")
        print(f"rows: {rows}, cols: {cols}")
    
    if number_of_images != number_of_labels:
        raise Exception('number of labels did not match the number of images')
    
    # Get the data and the labels
    X = np.zeros((number_of_labels, rows, cols), dtype=np.float32) # initalize np array
    y = np.zeros((number_of_labels, 1), dtype=np.uint8) # intalize empty array 
    
    for i in range(number_of_labels):
        if verbose and i%10000==0:
            print(f'processed {i} images')
        for row in range(rows):
            for col in range(cols):
                tmp_pixel = images.read(1) # read one byte
                tmp_pixel = unpack('>B', tmp_pixel)[0]
                X[i][row][col] = tmp_pixel
        tmp_label = labels.read(1) # read one byte
        tmp_label = unpack('>B', tmp_label)[0]
        y[i] = tmp_label
        
    images.close()
    labels.close()
    
    return X, y

if __name__ == '__main__':
    X, y = get_labeled_data('./data/train-images-idx3-ubyte.gz', './data/train-labels-idx1-ubyte.gz')
    print(X, y)
    