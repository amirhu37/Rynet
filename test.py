import os
import nnet
from nnet import nnet

import numpy as np

from tools import relu


linear_layer = nnet.Linear(in_features=3, out_features=2, is_bias=True)
linear_layer1 = nnet.Linear(in_features=3, out_features=2, is_bias=False)



class SimpleNN(nnet.Neuaral):
    def __init__(self):
        super(SimpleNN, self).__init__()

        self.fc1 = nnet.Linear(60, 12)  
        self.fc2 = nnet.Linear(12, 4)
        self.fc3 = nnet.Linear(4, 3) 
        pass

    def forward(self, x : np.ndarray):
        x = x.reshape(1,-1)
        x = relu(self.fc1(x))
        x = relu(self.fc2(x))
        x = relu(self.fc3(x))
        x = nnet.sigmoid(x)
        # x = sigmoid(x)
        return x
    
x = np.random.rand(20,3)

class custom_layer(nnet.Layer):
    def __init__(self, in_features, out_features, is_bias=True , *args, **kwargs ):
        self.in_features = in_features
        self.out_features = out_features
        # super(custom_layer, self).__init__()
        pass
    def su(self):
        print( self.in_features * self.out_features )


os.system("cls")
cls = SimpleNN()
# l = nnet.Layer()
print(linear_layer.__new__.__doc__)
lyer = custom_layer(2 , 30)
lyer.su()
print(lyer.__class__.__name__)
print(lyer.__dict__)

print(cls.__dict__.keys())
print(lyer.__dict__.keys()) 

param = cls.parameters()
pr  = param['fc1'].weights 
print(pr[1])
y = cls.forward(x)

print("--------------")
param = cls.parameters()
pr  = param['fc1'].weights 
print(pr[1])
print(y)