import os
import nnet
# import nnet.nnet.mlp
# import neural
import numpy as np

from tools import relu, sigmoid


linear_layer = nnet.Linear(in_features=3, out_features=2, is_bias=True)
linear_layer1 = nnet.Linear(in_features=3, out_features=2, is_bias=False)

# print("Linear layer created successfully:", linear_layer)
# print("Linear layer created successfully:", linear_layer1)


class SimpleNN(nnet.Mlp):
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
        x = sigmoid(x)
        return x
    
x = np.random.rand(20,3)

class ts:
    def __init__(self) -> None:
        self._1 = np.random.rand(2,1)
        self._2 = np.random.rand(2,1)
        self._3 = np.random.rand(2,1)
        pass


t = ts()
os.system("cls")
cls = SimpleNN()

# print(cls.__dict__)
# print(t.__getattribute__("__dict__"))
# print(cls.__getattribute__("__dict__"))

print(cls.__dict__.keys().__iter__()) 
param = cls.parameters()
# print(param)
pr  = param['fc1'].weights # = np.random.rand(2,3)
# print(pr[1])
# print("fx1", param['fc1'].weights)
y = cls.forward(x)

# print(y)