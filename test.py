import nnet
# import neural
import numpy as np

from tools import relu, sigmoid


linear_layer = nnet.Linear(in_features=3, out_features=2, is_bias=True)
linear_layer1 = nnet.Linear(in_features=3, out_features=2, is_bias=False)

print("Linear layer created successfully:", linear_layer)
print("Linear layer created successfully:", linear_layer1)


class SimpleNN(nnet.mlp):
    def __init__(self):
        super(SimpleNN, self).__init__()

        self.fc1 = nnet.Linear(6, 12)  
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


print(type(linear_layer.weights))
print(linear_layer)
print(linear_layer1.__dict__)

cls = SimpleNN()

x = np.random.rand(2,3)
y = cls.forward(x)
print(y)