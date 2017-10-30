from __future__ import division
import struct
import numpy as np
from NN_Dropout import NetworkClass


with open('MNIST_data/train-labels.idx1-ubyte', 'rb') as f:
    zero, data_type, dims = struct.unpack('>HBB', f.read(4))
    shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
    yTrain_original = np.fromstring(f.read(), dtype=np.uint8).reshape(shape)
    
with open('MNIST_data/train-images.idx3-ubyte', 'rb') as f:
    zero, data_type, dims = struct.unpack('>HBB', f.read(4))
    shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
    XTrain_original = np.fromstring(f.read(), dtype=np.uint8).reshape(shape)    

with open('MNIST_data/t10k-images-idx3-ubyte', 'rb') as f:
    zero, data_type, dims = struct.unpack('>HBB', f.read(4))
    shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
    XTest = np.fromstring(f.read(), dtype=np.uint8).reshape(shape)
    
with open('MNIST_data/t10k-labels.idx1-ubyte', 'rb') as f:
    zero, data_type, dims = struct.unpack('>HBB', f.read(4))
    shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
    yTest = np.fromstring(f.read(), dtype=np.uint8).reshape(shape)


n = XTrain_original.shape[0]
a = XTrain_original.shape[1]
b = XTrain_original.shape[2]

n_test = XTest.shape[0]
a_test = XTest.shape[1]
b_test = XTest.shape[2]

XTest = np.reshape(XTest, (n_test,a_test*b_test))

D_0 = 784
D_1 = 128
D_2 = 64
D_3 = 10

#Entire dataset
XTrain = np.reshape(XTrain_original, (n,a*b))
yTrain = np.array(np.zeros((yTrain_original.shape[0],D_3)))

# Multi-class classifier
for i in range(0,yTrain_original.shape[0]):
    yTrain[i,yTrain_original[i]]=1
    
NN = NetworkClass(D_0,D_1,D_2,D_3, n)
    
NN.train(XTrain, yTrain, XTest, yTest)
            
NN.results_plot()
