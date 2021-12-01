from typing_extensions import Annotated
import neural_net as ann
import numpy as np
import requests, gzip, os, hashlib

#fetch data
path='/home/reese/School/5523/mnist-from-numpy/data'
def fetch(url):
  fp = os.path.join(path, hashlib.md5(url.encode('utf-8')).hexdigest())
  if os.path.isfile(fp):
     with open(fp, "rb") as f:
        data = f.read()
  else:
     with open(fp, "wb") as f:
        data = requests.get(url).content
        f.write(data)
  return np.frombuffer(gzip.decompress(data), dtype=np.uint8).copy()

X = fetch("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
Y = fetch("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz")[8:]
X_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28*28))
Y_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz")[8:]



#Validation split
rand=np.arange(60000)
np.random.shuffle(rand)
train_no=rand[:50000]
val_no=np.setdiff1d(rand,train_no)
X_train,X_val=X[train_no,:,:],X[val_no,:,:]
Y_train,Y_val=Y[train_no],Y[val_no]


layer_sizes = (784, 128, 10)

net = ann.NeuralNetwork(layer_sizes)
net.train(X_train, Y_train, X_val, Y_val)
net.test(X_test, Y_test)

