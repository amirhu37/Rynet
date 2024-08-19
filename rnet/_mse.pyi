from typing import Any
# from loss import *

__all__ = __doc__

class MSELoss:
    def __init__(self, reduction : str = "mean") -> None:...
    def __call__(self, *args: Any, **kwds: Any) -> Any:...