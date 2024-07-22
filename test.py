import os
from nnet import nnet
# from nnet import layers ,  
from nnet.nnet import * 
# print(nnet.__dict__)
import numpy as np

from tools import relu


linear_layer = Linear(in_features=3, out_features=2, is_bias=True)
linear_layer1 = Linear(in_features=3, out_features=2, is_bias=False)



class SimpleNN(Neuaral):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = Linear(3, 12)  
        self.fc2 = Linear(12, 4)
        self.fc3 = Linear(4, 1) 
        pass

    def forward(self, x : np.ndarray):
        # x = x.reshape(1,-1)
        x = relu(self.fc1(x))
        x = relu(self.fc2(x))
        # x = relu(self.fc3(x))
        # x = nnet.softmax(x)[0]
        # x = sigmoid(x)
        print(x.shape)
        x = self.fc3(x)
        return x
    
x = np.random.rand(20,3)
# y = np.random.randint(0,3,20)
y = np.random.rand(20)
print(y)
# y_c = np.eye(20,3)[y]
# print(y_c)


class custom_layer(Layer):
    def __init__(self, in_features, out_features, is_bias=True , *args, **kwargs ):
        self.in_features = in_features
        self.out_features = out_features
        # super(custom_layer, self).__init__()
        pass
    def su(self):
        print( self.in_features * self.out_features )


# os.system("cls")
cls = SimpleNN()

param = cls.parameters()
pr  = param['fc1'].weights 
print(x[0])
y_hat = cls.forward(x[0])
print(f"y {y[0]}")
print(f"y_hat {y_hat}")
criterion = MSELoss("sum")
loss = criterion(np.max(y_hat) , y[0])
print(loss)
