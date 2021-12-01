import numpy as np
import requests, gzip, os, hashlib



class NeuralNetwork:

    def __init__(self, layer_sizes):
        self.weights = [[], []]
        self.weights[0] = self.randomize(layer_sizes[0], layer_sizes[1])
        self.weights[1] = self.randomize(layer_sizes[1], layer_sizes[2])
        
        print(np.shape(self.weights[0]), np.shape(self.weights[1]))
        self.learning_rate = 0.001
        self.epochs = 10000
        self.batch = 128
        self.losses,self.accuries,self.val_accuracies=[],[],[]
    #sigmoid function
    def sigmoid(self, x):
        return 1/(np.exp(-x)+1)    

    def randomize(self, x,y):
        layer=np.random.uniform(-1.,1.,size=(x,y))/np.sqrt(x*y)
        return layer.astype(np.float32)

    #derivative of sigmoid
    def d_sigmoid(self, x):
        return (np.exp(-x))/((np.exp(-x)+1)**2)

    #softmax function
    def softmax(self, x):
        exp_element=np.exp(x-x.max())
        return exp_element/np.sum(exp_element,axis=0)

    #derivative of softmax
    def d_softmax(self, x):
        exp_element=np.exp(x-x.max())
        return exp_element/np.sum(exp_element,axis=0)*(1-exp_element/np.sum(exp_element,axis=0))

    #forward and backward pass
    def forward_backward_pass(self, x,y):
        targets = np.zeros((len(y),10), np.float32)
        targets[range(targets.shape[0]),y] = 1

        x_l1=x.dot(self.weights[0])
        x_sigmoid=self.sigmoid(x_l1)
        x_l2=x_sigmoid.dot(self.weights[1])
        out=self.softmax(x_l2)
        error=2*(out-targets)/out.shape[0]*(self.d_softmax(x_l2))
        update_l2=x_sigmoid.T@error
        
        
        error=((self.weights[1]).dot(error.T)).T*self.d_sigmoid(x_l1)
        update_l1=x.T@error

        
        return out, update_l1, update_l2

    def train(self, X_train, Y_train, X_val, Y_val):
        for i in range(self.epochs):
            #randomize and create batches
            sample=np.random.randint(0,X_train.shape[0],size=(self.batch))
            x=X_train[sample].reshape((-1,28*28))
            y=Y_train[sample]

            out, update_l1, update_l2 = NeuralNetwork.forward_backward_pass(self, x,y)   
            category=np.argmax(out,axis=1)
            
            accuracy=(category==y).mean()
            self.accuries.append(accuracy.item())
            
            loss=((category-y)**2).mean()
            self.losses.append(loss.item())

            self.weights[0] = self.weights[0] - self.learning_rate * update_l1
            self.weights[1] = self.weights[1] - self.learning_rate * update_l2

            #testing our model using the validation set every 30 epochs
            if(i%30==0):    
                X_val=X_val.reshape((-1,28*28))
                val_out=np.argmax(self.softmax(self.sigmoid(X_val.dot(self.weights[0])).dot(self.weights[1])),axis=1)
                val_acc=(val_out==Y_val).mean()
                self.val_accuracies.append(val_acc.item())
            if(i%500==0): 
                print(f'For {i}th epoch: train accuracy: {accuracy:.3f}| validation accuracy:{val_acc:.3f}')

    def test(self, X_test, Y_test):
        test_out=np.argmax(self.softmax(self.sigmoid(X_test.dot(self.weights[0])).dot(self.weights[1])),axis=1)
        test_acc=(test_out==Y_test).mean().item()
        print(f'Test accuracy = {test_acc:.4f}')
