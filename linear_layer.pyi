import layers

class Linear:
    in_feature : int
    out_feature : int
    bias : bool = True
    def __init__(self, in_feature : int, out_feature : int, bias : bool =True):
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.bias = bias
        ...
    def parameters()-> dict:...
