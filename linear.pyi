# linear.pyi
from typing import Any, Optional, Tuple
from nn.layer import Layer

class Linear(Layer):
    weights: Any
    bias: Any
    is_bias: bool
    trainable: bool
    shape: Tuple[int, int]

    def __new__(cls, in_features: int, out_features: int, is_bias: Optional[bool] = ..., trainable: Optional[bool] = ..., *args: Optional[Any] , **kwargs: Optional[Any]) -> 'Linear': ...
