import os
# import rnet
from rntet import linear as lr
# from rntet import Linear, Neuaral, Layer, MSELoss
# from rnet.layer import Linear
# from rnet.nn import Neuaral

# from rnet.rnet import MSELoss 
# print(rnet.__dict__)
import numpy as np
from rntet import Tensor
from rntet.layer import Layer
from rntet.neural import Neuaral
from rntet.loss import MSMSELoss
from tools import relu


linear_layer = lr.Linear(3, 2,True)
# print(linear_layer.__dict__)
linear_layer1 = lr.Linear(in_features=3, out_features=2, is_bias=False)



class SimpleNN(Neuaral):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = lr.Linear(3, 12)  
        self.fc2 = lr.Linear(12, 4)
        self.fc3 = lr.Linear(4, 3) 
        pass

    def forward(self, x : np.ndarray):
        # x = x.reshape(1,-1)
        x = relu(self.fc1(x))
        x = relu(self.fc2(x))
        x = relu(self.fc3(x))
        print(x.shape)
        return x

class custom_layer(Layer):
    def __init__(self, in_features, out_features, is_bias=True , *args, **kwargs ):
        self.in_features = in_features
        self.out_features = out_features
        # super(custom_layer, self).__init__()
        pass
    def some_custom_method(self):
        print( self.in_features * self.out_features )
        pass



x = np.random.rand(20,3)

tensor = Tensor(x)
print(tensor)
print(type(tensor))


y = np.random.randint(0,3,20)
print(y)
y_c = np.eye(20,3)[y]
print(y_c)




# os.system("cls")
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
print(param)

# criterion = MSELoss('mean')
# print(
#     y_hat  ,
#     y_c[0]
# )
# loss = criterion(y_hat[0] , y_c[0])
# print(loss)

# for _ in range(10):
#     y_hat = cls(x[0])
#     criterion = MSELoss('mean')
#     loss = criterion(y_hat[0] , y_c[0])
    
# print(loss)