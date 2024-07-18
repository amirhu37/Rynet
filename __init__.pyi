from nnet import *

class Linear:
    def __init__(self, in_feature, out_feature, bias=True):
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.bias = bias
        ...
    def parameters()-> dict:...

class mlp:
    def __init__(self, layers, activation='relu', dropout=0.0):
        self.layers = layers
        self.activation = activation
        self.dropout = dropout
        pass
    