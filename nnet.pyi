import numpy as np
# from linear_layer import Linear
# from neural_net import Neuaral
# from nnet import nnet


class Linear:
    in_feature : int
    out_feature : int
    bias : bool = True
    def __init__(self, in_feature : int, out_feature : int, bias : bool =True):
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.bias = bias
        ...



class Neuaral:    
    def __init__(self, *args, **kwargs) -> None: ...
    def forward(self, x: np.ndarray) -> np.ndarray: ...