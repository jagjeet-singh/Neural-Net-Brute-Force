from __future__ import division
from scipy.special import expit
import struct
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

learning_rate = 0.001
iterations = 50
minibatch_size = 10000
gamma_BN_1 = 1
beta_BN_1 = 0
gamma_BN_2 = 1
beta_BN_2 = 0
gamma_BN_3 = 1
beta_BN_3 = 0
smoothing_factor = np.exp(-8)

#Entire dataset
XTrain_complete = np.reshape(XTrain_original, (n,a*b))
yTrain_complete = np.array(np.zeros((yTrain_original.shape[0],D_3)))

# Multi-class classifier
for i in range(0,yTrain_original.shape[0]):
    yTrain_complete[i,yTrain_original[i]]=1

#Weights and biases between input layer and first hidden layer
W1 = np.array(np.linspace(-0.01,0.01,num=D_0*D_1), dtype=np.float64)
W1 = np.reshape(W1,(D_1,D_0))
b1 = np.array(np.zeros((1,D_1)))

#Weights and biases between first hidden layer and second hidden layer
W2 = np.array(np.linspace(-0.01,0.01,num=D_1*D_2), dtype=np.float64)
W2 = np.reshape(W2,(D_2,D_1))
b2 = np.array(np.zeros((1,D_2)))

#Weights and biases between second hidden layer and output layer
W3 = np.array(np.linspace(-0.01,0.01,num=D_2*D_3), dtype=np.float64)
W3 = np.reshape(W3,(D_3,D_2))
b3 = np.array(np.zeros((1,D_3))) 

Err = np.ones((iterations))
Accuracy = np.ones((iterations))
itern_axis = np.array(np.linspace(1,iterations,num=iterations))
   
for iter in range(1,iterations+1):
    
    Err_mini = np.ones((int(n/minibatch_size)))
    Accuracy_mini = np.ones((int(n/minibatch_size)))
    
    for mini_iter in range(0,int(n/minibatch_size)):
        
        XTrain = np.array(np.zeros((minibatch_size,D_0)))
        yTrain = np.array(np.zeros((minibatch_size,D_3)))
        mini_start_index = int(mini_iter*minibatch_size)
        mini_end_index = int(mini_iter*minibatch_size+minibatch_size)
        
        XTrain = XTrain_complete[mini_start_index:mini_end_index,:]
        yTrain = yTrain_complete[mini_start_index:mini_end_index,:]
          
        z1 = XTrain.dot(W1.transpose())+b1
        
        mean_1 = np.array(np.mean(z1,axis=0))
        variance_1 = np.var(z1, axis=0)
      
        
        u1 = (z1 - mean_1)/np.sqrt(variance_1+smoothing_factor)
        z1_hat = gamma_BN_1*u1+beta_BN_1
        y1 = expit(z1_hat)
        
        
        z2 = y1.dot(W2.transpose())+b2
        mean_2 = np.array(np.mean(z2,axis=0))
        variance_2 = np.var(z2, axis=0)
        
        u2 = (z2 - mean_2)/np.sqrt(variance_2+smoothing_factor)
        z2_hat = gamma_BN_2*u2+beta_BN_2

        y2 = expit(z2_hat)
        
        z3 = y2.dot(W3.transpose())+b3
        mean_3 = np.array(np.mean(z3,axis=0))
        variance_3 = np.var(z3, axis=0)
        
        u3 = (z3 - mean_3)/np.sqrt(variance_3+smoothing_factor)
        z3_hat = gamma_BN_3*u3+beta_BN_3
        y3 = expit(z3_hat)
        
#        print(np.sum(W3,axis=1))
        
        Err_mini[mini_iter] = np.sum(np.sum(np.multiply(-yTrain,np.log(y3))+np.multiply(-(1-yTrain),np.log((1-y3)))))/minibatch_size
        print(str(iter)+"  ::  "+ str(mini_iter)+"  ::  "+str(Err_mini[mini_iter]))
        
        Accuracy_mini[mini_iter] = acc.getAccuracy(XTest,yTest,W1,W2,W3,b1,b2,b3)
        print(str(iter)+"  ::  "+ str(mini_iter)+"  ::  "+str(Accuracy_mini[mini_iter]))
            
        Delta_Err_by_y3 = np.divide(-yTrain,y3)+np.divide(1-yTrain,1-y3)
        Delta_Err_by_z3_hat = np.multiply(Delta_Err_by_y3,np.multiply(y3,1-y3))
        Delta_Err_by_gamma_3 = np.sum(u3*Delta_Err_by_z3_hat,axis=0)
        Delta_Err_by_beta_3 = np.sum(Delta_Err_by_z3_hat,axis=0)
        Delta_Err_by_u_3 = gamma_BN_3*Delta_Err_by_z3_hat
        Delta_Err_by_variance_3 = np.sum((z3-mean_3)*Delta_Err_by_u_3,axis=0)
        Delta_Err_by_variance_3 = Delta_Err_by_variance_3*((variance_3+smoothing_factor)**(-1.5))*(-0.5)
        Delta_Err_by_mean_3 = np.sum((-1/(np.sqrt(variance_3+smoothing_factor)))*Delta_Err_by_u_3,axis=0)
        Delta_Err_by_mean_3 = Delta_Err_by_mean_3 + np.sum(z3-mean_3,axis=0)*(-2)*Delta_Err_by_variance_3/minibatch_size
        Delta_Err_by_z3 = Delta_Err_by_u_3/np.sqrt(variance_3+smoothing_factor)+Delta_Err_by_variance_3*2*(z3-mean_3)/minibatch_size
        Delta_Err_by_z3 = Delta_Err_by_z3+Delta_Err_by_mean_3/minibatch_size
        Delta_Err_by_w3 = Delta_Err_by_z3.transpose().dot(y2)/minibatch_size
        Delta_Err_by_b3 = np.sum(Delta_Err_by_z3,axis=0)/minibatch_size
        
        Delta_Err_by_y2 = Delta_Err_by_z3.dot(W3)
        Delta_Err_by_z2_hat = np.multiply(Delta_Err_by_y2,np.multiply(y2,1-y2))
        Delta_Err_by_gamma_2 = np.sum(u2*Delta_Err_by_z2_hat,axis=0)
        Delta_Err_by_beta_2 = np.sum(Delta_Err_by_z2_hat,axis=0)
        Delta_Err_by_u_2 = gamma_BN_2*Delta_Err_by_z2_hat
        Delta_Err_by_variance_2 = np.sum((z2-mean_2)*Delta_Err_by_u_2,axis=0)
        Delta_Err_by_variance_2 = Delta_Err_by_variance_2*((variance_2+smoothing_factor)**(-1.5))*(-0.5)
        Delta_Err_by_mean_2 = np.sum((-1/(np.sqrt(variance_2+smoothing_factor)))*Delta_Err_by_u_2,axis=0)
        Delta_Err_by_mean_2 = Delta_Err_by_mean_2 + np.sum(z2-mean_2,axis=0)*(-2)*Delta_Err_by_variance_2/minibatch_size
        Delta_Err_by_z2 = Delta_Err_by_u_2/np.sqrt(variance_2+smoothing_factor)+Delta_Err_by_variance_2*2*(z2-mean_2)/minibatch_size
        Delta_Err_by_z2 = Delta_Err_by_z2+Delta_Err_by_mean_2/minibatch_size
        Delta_Err_by_w2 = Delta_Err_by_z2.transpose().dot(y1)/minibatch_size
        Delta_Err_by_b2 = np.sum(Delta_Err_by_z2,axis=0)/minibatch_size
            
        Delta_Err_by_y1 = Delta_Err_by_z2.dot(W2)
        Delta_Err_by_z1_hat = np.multiply(Delta_Err_by_y1,np.multiply(y1,1-y1))
        Delta_Err_by_gamma_1 = np.sum(u1*Delta_Err_by_z1_hat, axis=0)
        Delta_Err_by_beta_1 = np.sum(Delta_Err_by_z1_hat, axis=0)
        Delta_Err_by_u_1 = gamma_BN_1*Delta_Err_by_z1_hat
        Delta_Err_by_variance_1 = np.sum((z1-mean_1)*Delta_Err_by_u_1,axis=0)
        Delta_Err_by_variance_1 = Delta_Err_by_variance_1*((variance_1+smoothing_factor)**(-1.5))*(-0.5)
        Delta_Err_by_mean_1 = np.sum((-1/(np.sqrt(variance_1+smoothing_factor)))*Delta_Err_by_u_1,axis=0)
        Delta_Err_by_mean_1 = Delta_Err_by_mean_1 + np.sum(z1-mean_1,axis=0)*(-2)*Delta_Err_by_variance_1/minibatch_size
        Delta_Err_by_z1 = Delta_Err_by_u_1/np.sqrt(variance_1+smoothing_factor)+Delta_Err_by_variance_1*2*(z1-mean_1)/minibatch_size
        Delta_Err_by_z1 = Delta_Err_by_z1+Delta_Err_by_mean_1/minibatch_size
        Delta_Err_by_w1 = Delta_Err_by_z1.transpose().dot(XTrain)/minibatch_size
        Delta_Err_by_b1 = np.sum(Delta_Err_by_z1,axis=0)/minibatch_size
        
        W1 = W1-learning_rate*Delta_Err_by_w1
        gamma_BN_1 = gamma_BN_1 - learning_rate*Delta_Err_by_gamma_1/minibatch_size
        beta_BN_1 = beta_BN_1 - learning_rate*Delta_Err_by_beta_1/minibatch_size
        W2 = W2-learning_rate*Delta_Err_by_w2
        gamma_BN_2 = gamma_BN_2 - learning_rate*Delta_Err_by_gamma_2/minibatch_size
        beta_BN_2 = beta_BN_2 - learning_rate*Delta_Err_by_beta_2/minibatch_size
        W3 = W3-learning_rate*Delta_Err_by_w3
        gamma_BN_3 = gamma_BN_3 - learning_rate*Delta_Err_by_gamma_3/minibatch_size
        beta_BN_3 = beta_BN_3 - learning_rate*Delta_Err_by_beta_3/minibatch_size
        
        b1 = b1-learning_rate*Delta_Err_by_b1
        b2 = b2-learning_rate*Delta_Err_by_b2
        b3 = b3-learning_rate*Delta_Err_by_b3
        
        
    
    Err[iter-1] = Err_mini.mean()
    Accuracy[iter-1]=Accuracy_mini.mean()


fig, ax=plt.subplots()
ax.plot(itern_axis,Accuracy,'-b.',label='lr = 0.1, batch size = 500')
plt.ylabel('Accuracy')
plt.xlabel('iteration')
ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:.2f}"))
legend = ax.legend(loc='upper center', shadow=True)
plt.ylim((0,100))
#plt.savefig('Accuracy_itern_mini_500.png')
plt.show()

#plt.plot(itern_axis,Err)
#plt.ylabel('Error')
#plt.xlabel('iteration')
#plt.savefig('Err_mini_500.png')
#plt.show()
