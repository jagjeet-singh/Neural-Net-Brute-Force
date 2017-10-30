from __future__ import division
from scipy.special import expit
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker

class NetworkClass:
    
    def __init__(self, D_0, D_1, D_2, D_3, n, learning_rate=0.001,iterations=50 ,minibatch_size=500 ,dropout_rate = 0.5):
        '''
        Initializes Parameters of the  Logistic Regression Model
        '''
        #Weights and biases between input layer and first hidden layer
        self.W1 = np.array(np.linspace(-0.01,0.01,num=D_0*D_1), dtype=np.float64)
        self.W1 = np.reshape(self.W1,(D_1,D_0))
        self.b1 = np.array(np.zeros((1,D_1)))
        
        #Weights and biases between first hidden layer and second hidden layer
        self.W2 = np.array(np.linspace(-0.01,0.01,num=D_1*D_2), dtype=np.float64)
        self.W2 = np.reshape(self.W2,(D_2,D_1))
        self.b2 = np.array(np.zeros((1,D_2)))
        
        #Weights and biases between second hidden layer and output layer
        self.W3 = np.array(np.linspace(-0.01,0.01,num=D_2*D_3), dtype=np.float64)
        self.W3 = np.reshape(self.W3,(D_3,D_2))
        self.b3 = np.array(np.zeros((1,D_3))) 
        
        # Hyper parameters
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.minibatch_size = minibatch_size
        self.dropout_rate = dropout_rate
        
        #Training Error
        self.Err = np.ones((iterations))
        
        #Test Accuracy
        self.Accuracy = np.ones((iterations))
        
        #Number of training examples
        self.n = n

    def get_training_error(self,yTrain, y3):
        
        #Cross Entropy Loss
        Err = np.sum(np.sum(np.multiply(-yTrain,np.log(y3))+np.multiply(-(1-yTrain),np.log((1-y3)))))/self.minibatch_size
        return Err          
        
    def get_test_accuracy(self, XTest, yTest):
        
        z1 = XTest.dot(self.W1.transpose())+self.b1
        y1 = expit(z1)
        z2 = y1.dot(self.W2.transpose())+self.b2
        y2 = expit(z2)
        z3 = y2.dot(self.W3.transpose())+self.b3
        y3 = expit(z3)

        y_hat = np.argmax(y3,axis=1)
    
        correct = np.sum(y_hat==yTest)
        accuracy = correct/np.size(yTest)*100
    
        return accuracy
        
    def train(self, XTrain, yTrain, XTest, yTest):

        # Training Error and Test accuracy after each minibatch
         Err_mini = np.ones((int(self.n/self.minibatch_size)))
         Accuracy_mini = np.ones((int(self.n/self.minibatch_size))) 
           
         for iter in range(1,self.iterations+1):
        
            for mini_iter in range(0,int(self.n/self.minibatch_size)):
                    # Forward Prop
                    
                XTrain_mini = np.array(np.zeros((self.minibatch_size,np.shape(XTrain)[1])))
                yTrain_mini = np.array(np.zeros((self.minibatch_size,np.shape(yTrain)[1])))
                mini_start_index = int(mini_iter*self.minibatch_size)
                mini_end_index = int(mini_iter*self.minibatch_size+self.minibatch_size-1)
                
                XTrain_mini = XTrain[mini_start_index:mini_end_index,:]
                yTrain_mini = yTrain[mini_start_index:mini_end_index,:]
                z1 = XTrain_mini.dot(self.W1.transpose())+self.b1
                y1 = expit(z1)/(1-self.dropout_rate)
                r1 = np.random.binomial(1, 1 - self.dropout_rate , size=y1.shape)
                y1 = y1*r1
                z2 = y1.dot(self.W2.transpose())+self.b2
                y2 = expit(z2)/(1-self.dropout_rate)
                r2 = np.random.binomial(1, 1 - self.dropout_rate , size=y2.shape)
                y2 = y2*r2
                z3 = y2.dot(self.W3.transpose())+self.b3
                y3 = expit(z3)
                #Error and accuracy after forward prop
            
                Err_mini[mini_iter] = self.get_training_error(yTrain_mini, y3)
                print(str(iter)+"  ::  "+ str(mini_iter)+"  ::  "+str(Err_mini[mini_iter]))
                
                Accuracy_mini[mini_iter] = self.get_test_accuracy(XTest,yTest)
                print(str(iter)+"  ::  "+ str(mini_iter)+"  ::  "+str(Accuracy_mini[mini_iter]))
                
                
                #Backprop
                
                Delta_Err_by_y3 = np.divide(-yTrain_mini,y3)+np.divide(1-yTrain_mini,1-y3)
                Delta_Err_by_z3 = np.multiply(Delta_Err_by_y3,np.multiply(y3,1-y3))
                Delta_Err_by_w3 = Delta_Err_by_z3.transpose().dot(y2*r2)/self.minibatch_size
                Delta_Err_by_b3 = np.sum(Delta_Err_by_z3,axis=0)/self.minibatch_size
                
                Delta_Err_by_y2 = Delta_Err_by_z3.dot(self.W3)
                Delta_Err_by_z2 = np.multiply(Delta_Err_by_y2,np.multiply(y2,1-y2))*r2
                Delta_Err_by_w2 = Delta_Err_by_z2.transpose().dot(y1*r1)/self.minibatch_size
                Delta_Err_by_b2 = np.sum(Delta_Err_by_z2,axis=0)/self.minibatch_size
                    
                Delta_Err_by_y1 = Delta_Err_by_z2.dot(self.W2)
                Delta_Err_by_z1 = np.multiply(Delta_Err_by_y1,np.multiply(y1,1-y1))*r1
                Delta_Err_by_w1 = Delta_Err_by_z1.transpose().dot(XTrain_mini)/self.minibatch_size
                Delta_Err_by_b1 = np.sum(Delta_Err_by_z1,axis=0)/self.minibatch_size
                
                self.W1 = self.W1-self.learning_rate*Delta_Err_by_w1
                self.W2 = self.W2-self.learning_rate*Delta_Err_by_w2
                self.W3 = self.W3-self.learning_rate*Delta_Err_by_w3
                
                self.b1 = self.b1-self.learning_rate*Delta_Err_by_b1
                self.b2 = self.b2-self.learning_rate*Delta_Err_by_b2
                self.b3 = self.b3-self.learning_rate*Delta_Err_by_b3
    
            self.Err[iter-1] = Err_mini.mean()
            self.Accuracy[iter-1]=Accuracy_mini.mean()

    def results_plot(self):
        
        
        itern_axis = np.array(np.linspace(1,self.iterations,num=self.iterations))
        
        fig, ax=plt.subplots()
        ax.plot(itern_axis,self.Accuracy,'-b.',label='lr = 0.001, batch size = 500')
        plt.ylabel('Accuracy')
        plt.xlabel('iteration')
        plt.title('Test Accuracy for Dropout')
        ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:.2f}"))
        legend = ax.legend(loc='upper center', shadow=True)
        plt.ylim((0,100))
        plt.savefig('Accuracy_itern_dropout_mini_500.png')
        plt.show()
        
        fig, ax=plt.subplots()
        ax.plot(itern_axis, self.Err,'-b.',label='lr = 0.001, batch size = 500')
        plt.ylabel('Error')
        plt.xlabel('iteration')
        plt.title('Training Error for Dropout')
        ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:.2f}"))
        legend = ax.legend(loc='upper center', shadow=True)
        plt.ylim((0,10))
        plt.savefig('Error_itern_dropout_mini_500.png')
        plt.show()
        
