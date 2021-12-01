import numpy as np


class NeuralNetwork:

    def __init__(self, layer_sizes):
        weight_shapes = [(a,b) for a,b in zip(layer_sizes[1:],layer_sizes[:-1])]
        self.weights = [np.random.standard_normal(s)/s[1]**.5 for s in weight_shapes]
        self.biases = [np.zeros((s, 1)) for s in layer_sizes[1:]]
        self.learning_rate = 0.001
        self.epochs = 10000
        self.batch = 128
        self.losses,self.accuries,self.val_accuracies=[],[],[]
    #sigmoid function
    def sigmoid(x):
        return 1/(np.exp(-x)+1)    

    #derivative of sigmoid
    def d_sigmoid(x):
        return (np.exp(-x))/((np.exp(-x)+1)**2)

    #softmax function
    def softmax(x):
        exp_element=np.exp(x-x.max())
        return exp_element/np.sum(exp_element,axis=0)

    #derivative of softmax
    def d_softmax(x):
        exp_element=np.exp(x-x.max())
        return exp_element/np.sum(exp_element,axis=0)*(1-exp_element/np.sum(exp_element,axis=0))

    #forward and backward pass
    def forward_backward_pass(x,y):
        targets = np.zeros((len(y),10), np.float32)
        targets[range(targets.shape[0]),y] = 1
    
        
        x_l1=x.dot(l1)
        x_sigmoid=sigmoid(x_l1)
        x_l2=x_sigmoid.dot(l2)
        out=softmax(x_l2)
    
    
        error=2*(out-targets)/out.shape[0]*d_softmax(x_l2)
        update_l2=x_sigmoid.T@error
        
        
        error=((l2).dot(error.T)).T*d_sigmoid(x_l1)
        update_l1=x.T@error

        self.weights[0] = self.learning_rate * update_l1
        self.weights[1] = self.learning_rate * update_l2
        return out

    def train(self, X_train, Y_train):
        for i in range(self.epochs):
            #randomize and create batches
            sample=np.random.randint(0,X_train.shape[0],size=(self.batch))
            x=X_train[sample].reshape((-1,28*28))
            y=Y_train[sample]

            out = forward_backward_pass(x,y)   
            category=np.argmax(out,axis=1)
            
            accuracy=(category==y).mean()
            self.accuries.append(accuracy.item())
            
            loss=((category-y)**2).mean()
            self.losses.append(loss.item())

            #testing our model using the validation set every 30 epochs
            if(i%30==0):    
                X_val=X_val.reshape((-1,28*28))
                val_out=np.argmax(softmax(sigmoid(X_val.dot(l1)).dot(l2)),axis=1)
                val_acc=(val_out==Y_val).mean()
                self.val_accuracies.append(val_acc.item())
            if(i%1000==0): 
                print(f'For {i}th epoch: train accuracy: {accuracy:.3f}| validation accuracy:{val_acc:.3f}')

    def predict(self, a):
        for w, b in zip(self.weights, self.biases):
            z = np.matmul(w, a) + b
            print(z[0])
            a = self.activation(np.matmul(w, a) + b)
        return a

    def print_accuracy(self, images, labels):
        predictions = self.predict(images)
        num_correct = sum([np.argmax(a) == np.argmax(b)
                          for a, b in zip(predictions, labels)])
        print('{0}/{1} accuracy:{2}%'.format(num_correct, len(images),))

    @staticmethod
    def activation(x):
        return 1/(1-np.exp(-x))
