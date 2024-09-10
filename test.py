import os
import numpy as np
from tools import relu


# import rnet
# from rnet import rnet
import rynet.rynet as rnet
from rynet.rynet import nn
# from rnet.rnet import nn
# from rnet import nn
# import  rnet.nn as nn

# import  rnet.rnet as rnt
# from rnet.rnet import rnt
# import rnet as nn
# from rnet import rnet
# from rnet import  nn

# print(
#     rnet. __dict__    )

print(
    rnet.nn.__dict__    )


# print(
#     rnet.Tensor
# )

print("="*50)
linear_layer = nn. Linear(in_features=3, out_features=2, is_bias=True)
print(linear_layer)


# class SimpleNN(nn.Model):
#     def __init__(self):
#         # super(SimpleNN, self).__init__()
#         self.fc1 = nn.Linear(3, 12)  
#         self.fc2 = nn.Linear(12, 4)
#         self.fc3 = nn.Linear(4, 3) 
#         pass

#     def forward(self, x : np.ndarray):
#         # super().forward()
#         x = relu(self.fc1(x))
#         x = relu(self.fc2(x))
#         x = relu(self.fc3(x))
#         print(x.shape)
#         return x
    

# # class custom_layer(rn.Layer):
# #     def __init__(self, in_features, out_features, is_bias=True , *args, **kwargs ):
# #         self.in_features = in_features
# #         self.out_features = out_features
# #         # super(custom_layer, self).__init__()
# #         pass
# #     def some_custom_method(self):
# #         print( self.in_features * self.out_features )
# #         pass




# x = np.random.rand(20,3)
# # y = np.random.randint(0,3,20)
# y = np.random.randint(0,3,20)
# # print(y)
# y_c = np.eye(20,3)[y]
# # print(y_c)
# # print("y_c", y_c)

# # tensor = rnet.Tensor(x)
# # print("TT ",x)


# linear_layer = nn.Linear(in_features=3, out_features=2, is_bias=True)
# print("linear_layer",linear_layer.weight)

# linear_layer1 = nn.Linear(in_features=3, out_features=2, is_bias=False)

# print("linear_layer",linear_layer.weight)

# # print("linear_layer 1 ",linear_layer.forward(x))
# # print("TT: ",linear_layer)

# cls = SimpleNN()
# y_hat = cls(x[0])

# print(f"""
# x.0 
# {x[0]}
# ---------------
# y_hat.0
# {y_hat}
# """)

# param = cls.parameters()
# print("param ",param, )
# # criterion = MSELoss('mean')

# # loss = criterion(y_hat[0] , y_c[0])
# # print(loss)

# # for _ in range(10):
# #     y_hat = cls(x[0])
# #     criterion = MSELoss('mean')
# #     loss = criterion(y_hat[0] , y_c[0])
    
# # print(loss)