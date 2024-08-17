# from rntet import nn
from typing import Any, Optional, Self
from numpy import ndarray
# from rnet.nn import Linear

class Linear:
    in_features : int
    out_features : int
    is_bias : bool = True
    trainable: bool = True
    def __new__(cls, in_features: int, 
            out_features: int, 
            is_bias: Optional[bool] = ..., 
            trainable: Optional[bool] = ..., 
            *args: Optional[Any] , 
            **kwargs: Optional[Any])->Self: ...
    def __init__(self, in_features : int, out_features : int, is_bias : bool =True, trainable: bool = True):
        self.in_features = in_features
        self.out_features = out_features
        self.is_bias = is_bias
        self.trainable = trainable
        ...



class Neuaral:    

    def __init__(self, *args, **kwargs) -> None: ...
    def forward(self, x: ndarray) -> ndarray: ...
    def parameters(self) -> dict:...
    def __call__(self, *args: Any, **kwds: Any) -> Any:...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...


