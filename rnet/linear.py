class Linear:
    in_features : int
    out_features : int
    is_bias : bool = True
    trainable: bool = True
    def __init__(self, in_features : int, out_features : int, is_bias : bool =True, trainable: bool = True):
        self.in_features = in_features
        self.out_features = out_features
        self.is_bias = is_bias
        self.trainable = trainable
        ...

