from pylab import imshow, show, cm

from process_images import get_labeled_data

def view_image(image, label=""):
    print(f'label: {label}')
    imshow(image, cmap=cm.gray)
    show()
    
if __name__ == '__main__':
    X, y = get_labeled_data('./data/t10k-images-idx3-ubyte.gz', './data/t10k-labels-idx1-ubyte.gz')
    print('got data')
    view_image(X[0], y[0])i