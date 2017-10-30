from __future__ import division
from scipy.special import expit
#import scikit-learn
from sklearn.utils import shuffle
import struct
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker
import getAccuracy as acc


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

D_0 = 784
D_1 = 128
D_2 = 64
D_3 = 10

initial_learning_rate = 0.1
iterations = 50
minibatch_size = 500
dropout_rate_input = 0
dropout_rate = 0.5

gamma_rms = 0.7
smoothing_value = math.exp(-8)

XTrain_complete = np.reshape(XTrain_original, (n,a*b))
yTrain_complete = np.array(np.zeros((yTrain_original.shape[0],D_3)))

for i in range(0,yTrain_original.shape[0]):
    yTrain_complete[i,yTrain_original[i]]=1

W1 = np.array(np.linspace(-0.01,0.01,num=D_0*D_1), dtype=np.float64)
W1 = np.reshape(W1,(D_1,D_0))

W2 = np.array(np.linspace(-0.01,0.01,num=D_1*D_2), dtype=np.float64)
W2 = np.reshape(W2,(D_2,D_1))

W3 = np.array(np.linspace(-0.01,0.01,num=D_2*D_3), dtype=np.float64)
W3 = np.reshape(W3,(D_3,D_2))

b1 = np.array(np.zeros((1,D_1)))
b2 = np.array(np.zeros((1,D_2)))
b3 = np.array(np.zeros((1,D_3))) 

Err = np.ones((iterations))
Accuracy = np.ones((iterations))
itern_axis = np.array(np.linspace(1,iterations,num=iterations))

Avg_sqr_Delta_W3 =np.array(np.zeros((D_3,D_2)))
learning_rate_W3 = initial_learning_rate*np.array(np.ones((D_3,D_2)))

Avg_sqr_Delta_W2 =np.array(np.zeros((D_2,D_1)))
learning_rate_W2 = initial_learning_rate*np.array(np.ones((D_2,D_1)))

Avg_sqr_Delta_W1 =np.array(np.zeros((D_1,D_0)))
learning_rate_W1 = initial_learning_rate*np.array(np.ones((D_1,D_0)))

for iter in range(1,iterations+1):
    
    Err_mini = np.ones((int(n/minibatch_size)))
    Accuracy_mini = np.ones((int(n/minibatch_size)))
    
    for mini_iter in range(0,int(n/minibatch_size)):

        
        XTrain = np.array(np.zeros((minibatch_size,D_0)))
        yTrain = np.array(np.zeros((minibatch_size,D_3)))
        mini_start_index = int(mini_iter*minibatch_size)
        mini_end_index = int(mini_iter*minibatch_size+minibatch_size-1)
        
        XTrain = XTrain_complete[mini_start_index:mini_end_index,:]
        yTrain = yTrain_complete[mini_start_index:mini_end_index,:]
        
        XTrain, yTrain = shuffle(XTrain, yTrain, random_state=0)
#        
#        shuffle = np.arange(np.shape(XTrain)[0])
#        shuffle = np.random.shuffle(shuffle)
#        XTrain = XTrain[shuffle,:]
#        yTrain = yTrain[shuffle,:]
        
        r0 = np.random.binomial(1, 1 - dropout_rate_input , size=XTrain.shape)
        XTrain = XTrain*r0
        z1 = XTrain.dot(W1.transpose())+b1
        y1 = expit(z1)/(1-dropout_rate)
#        r1_random = np.array(np.random.rand(1,np.shape(y1)[1]))
#        r1 = (r1_random>(1-dropout_rate))
        r1 = np.random.binomial(1, 1 - dropout_rate , size=y1.shape)
        y1 = y1*r1
        z2 = y1.dot(W2.transpose())+b2
        y2 = expit(z2)/(1-dropout_rate)
#        r2_random = np.array(np.random.rand(1,np.shape(y2)[1]))
#        r2 = (r2_random>(1-dropout_rate))
        r2 = np.random.binomial(1, 1 - dropout_rate , size=y2.shape)
        y2 = y2*r2
        z3 = y2.dot(W3.transpose())+b3
        y3 = expit(z3)
            
#        z1 = XTrain.dot(W1.transpose())+b1
#        y1 = expit(z1)
#        z2 = y1.dot(W2.transpose())+b2
#        y2 = expit(z2)
#        z3 = y2.dot(W3.transpose())+b3
#        y3 = expit(z3)
        
        
        Err_mini[mini_iter] = np.sum(np.sum(np.multiply(-yTrain,np.log(y3))+np.multiply(-(1-yTrain),np.log((1-y3)))))/minibatch_size
        print(str(iter)+"  ::  "+ str(mini_iter)+"  ::  "+str(Err_mini[mini_iter]))
        
        Accuracy_mini[mini_iter] = acc.getAccuracy(XTest,yTest,W1,W2,W3,b1,b2,b3)
        print(str(iter)+"  ::  "+ str(mini_iter)+"  ::  "+str(Accuracy_mini[mini_iter]))
            
            
        Delta_Err_by_y3 = np.divide(-yTrain,y3)+np.divide(1-yTrain,1-y3)
        Delta_Err_by_z3 = np.multiply(Delta_Err_by_y3,np.multiply(y3,1-y3))
#        Delta_Err_by_w3 = Delta_Err_by_z3.transpose().dot(y2)
        Delta_Err_by_w3 = Delta_Err_by_z3.transpose().dot(y2*r2)
        Delta_Err_by_b3 = np.sum(Delta_Err_by_z3,axis=0)/minibatch_size
        
        Delta_Err_by_w3_square = np.square(Delta_Err_by_z3.transpose()).dot(np.square(y2))        
        
        Delta_Err_by_y2 = Delta_Err_by_z3.dot(W3)
        Delta_Err_by_z2 = np.multiply(Delta_Err_by_y2,np.multiply(y2,1-y2))*r2
#        Delta_Err_by_z2 = np.multiply(Delta_Err_by_y2,np.multiply(y2,1-y2))
#        Delta_Err_by_w2 = Delta_Err_by_z2.transpose().dot(y1)
        Delta_Err_by_b2 = np.sum(Delta_Err_by_z2,axis=0)/minibatch_size
        Delta_Err_by_w2 = Delta_Err_by_z2.transpose().dot(y1*r1)
        Delta_Err_by_w2_square = np.square(Delta_Err_by_z2.transpose()).dot(np.square(y1))  
            
        Delta_Err_by_y1 = Delta_Err_by_z2.dot(W2)
        Delta_Err_by_z1 = np.multiply(Delta_Err_by_y1,np.multiply(y1,1-y1))*r1
#        Delta_Err_by_z1 = np.multiply(Delta_Err_by_y1,np.multiply(y1,1-y1))
        Delta_Err_by_w1 = Delta_Err_by_z1.transpose().dot(XTrain)
        
        Delta_Err_by_b1 = np.sum(Delta_Err_by_z1,axis=0)/minibatch_size
        Delta_Err_by_w1_square = np.square(Delta_Err_by_z1.transpose()).dot(np.square(XTrain))
                        
#        Delta_W3_accumulated = Delta_W3_accumulated+Delta_Err_by_w3
#        Delta_W2_accumulated = Delta_W2_accumulated+Delta_Err_by_w2
#        Delta_W1_accumulated = Delta_W1_accumulated+Delta_Err_by_w1
        
        b1 = b1-initial_learning_rate*Delta_Err_by_b1
        b2 = b2-initial_learning_rate*Delta_Err_by_b2
        b3 = b3-initial_learning_rate*Delta_Err_by_b3
    
        print(np.mean(learning_rate_W3))
        
        Avg_sqr_Delta_W3 = (gamma_rms*Avg_sqr_Delta_W3+(1-gamma_rms)*np.square(Delta_Err_by_w3/minibatch_size))
#        Avg_sqr_Delta_W3 = (gamma_rms*Avg_sqr_Delta_W3+(1-gamma_rms)*Delta_Err_by_w3_square/minibatch_size)
        learning_rate_W3 = initial_learning_rate/np.sqrt(Avg_sqr_Delta_W3+smoothing_value) 
        W3 = W3-learning_rate_W3*Delta_Err_by_w3/minibatch_size
        
        print(np.mean(Avg_sqr_Delta_W3))
    
        Avg_sqr_Delta_W2 = (gamma_rms*Avg_sqr_Delta_W2+(1-gamma_rms)*np.square(Delta_Err_by_w2/minibatch_size))
#        Avg_sqr_Delta_W2 = (gamma_rms*Avg_sqr_Delta_W2+(1-gamma_rms)*np.square(Delta_Err_by_w2_square/minibatch_size))
        learning_rate_W2 = initial_learning_rate/np.sqrt(Avg_sqr_Delta_W2+smoothing_value) 
        W2 = W2-learning_rate_W2*Delta_Err_by_w2/minibatch_size
    
        Avg_sqr_Delta_W1 = (gamma_rms*Avg_sqr_Delta_W1+(1-gamma_rms)*np.square(Delta_Err_by_w1/minibatch_size))
#        Avg_sqr_Delta_W1 = (gamma_rms*Avg_sqr_Delta_W1+(1-gamma_rms)*np.square(Delta_Err_by_w1_square/minibatch_size))
        learning_rate_W1 = initial_learning_rate/np.sqrt(Avg_sqr_Delta_W1+smoothing_value) 
        W1 = W1-learning_rate_W1*Delta_Err_by_w1/minibatch_size
        
    Err[iter-1] = Err_mini.mean()
    Accuracy[iter-1]=Accuracy_mini.mean()



fig, ax=plt.subplots()
ax.plot(itern_axis,Accuracy,'-b.',label='lr = 0.1, batch size = 500')
plt.ylabel('Accuracy')
plt.xlabel('iteration')
ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:.2f}"))
plt.ylim((0,100))
#plt.savefig('Accuracy_mini_500_rms.png')
plt.show()

#plt.plot(itern_axis,Err)
#plt.ylabel('Error')
#plt.xlabel('iteration')
#plt.savefig('Err_mini_500.png')
#plt.show()
