"""Converts the raw MNIST data to a numpy format."""
import os
import struct
import numpy as np
from array import array as pyarray

def load_mnist(dataset="training", digits=np.arange(10), path=".", size=60000):
    """Loads the MNIST raw data."""
    # Courtesy of https://gist.github.com/mfathirirhas/f24d61d134b014da029a
    if dataset == "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset == "testing":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()

    ind = [k for k in range(size) if lbl[k] in digits]
    N = size #int(len(ind) * size/100.)
    images = np.zeros((N, rows, cols), dtype=np.uint8)
    labels = np.zeros((N, 1), dtype=np.int8)
    for i in range(N): #int(len(ind) * size/100.)):
        images[i] = np.array(img[ind[i]*rows*cols : (ind[i]+1)*rows*cols])\
            .reshape((rows, cols))
        labels[i] = lbl[ind[i]]
    labels = [label[0] for label in labels]
    return images, labels

def main():
    X_train, y_train = load_mnist("training")
    X_test, y_test = load_mnist("testing", size=10000)
    np.save("test_img.npy", X_test)
    np.save("test_lab.npy", y_test)
    np.save("train_img.npy", X_train)
    np.save("train_lab.npy", y_train)

if __name__=="__main__":
    main()
