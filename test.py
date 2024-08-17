import os

# from rnet import Linear, Neuaral, Layer 
# from layers import Layer
import rntet
from rntet import nn, layer, loss
print(rntet.__dict__)
# from rnet.rnet import MSELoss 
# print(nnet.__dict__)
import numpy as np

from tools import relu


linear_layer = nn.Linear(in_features=3, out_features=2, is_bias=True)
linear_layer1 = nn.Linear(in_features=3, out_features=2, is_bias=False)



class SimpleNN(nn.Neuaral):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(3, 12)  
        self.fc2 = nn.Linear(12, 4)
        self.fc3 = nn.Linear(4, 3) 
        pass

    def forward(self, x : np.ndarray):
        # x = x.reshape(1,-1)
        x = relu(self.fc1(x))
        x = relu(self.fc2(x))
        # x = relu(self.fc3(x))
        # x = nnet.softmax(x)[0]
        # x = sigmoid(x)
        x = relu(self.fc3(x))
        print(x.shape)
        return x
    
x = np.random.rand(20,3)
# y = np.random.randint(0,3,20)
y = np.random.randint(0,3,20)
print(y)
y_c = np.eye(20,3)[y]
print(y_c)


class custom_layer(layer.Layer):
    def __init__(self, in_features, out_features, is_bias=True , *args, **kwargs ):
        self.in_features = in_features
        self.out_features = out_features
        # super(custom_layer, self).__init__()
        pass
    def some_custom_method(self):
        print( self.in_features * self.out_features )
        pass

os.system("cls")
cls = SimpleNN()
y_hat = cls(x[0])
print(f"""
x.0 
{x[0]}
---------------
y_hat.0
{y_hat}
""")

param = cls.parameters()
print("param ",param)
criterion = loss.MSELoss('mean')
print(
    y_hat  ,
    y_c[0]
)
loss = criterion(y_hat[0] , y_c[0])
print(loss)

for _ in range(10):
    y_hat = cls(x[0])
    criterion = loss.MSELoss('mean')
    loss = criterion(y_hat[0] , y_c[0])
    
print(loss)