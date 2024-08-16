from typing import Any


class MSMSELoss:
    reduction : str = "mean"
    def __init__(self, reduction : str = "mean") -> None:
        self.reduction = reduction
    def __call__(self, *args: Any, **kwds: Any) -> Any:...