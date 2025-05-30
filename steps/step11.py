# %% [markdown]
# 複数の引数を取れるように拡張する

# %%
from collections.abc import Iterable
from typing import Union
import numpy as np


# %%
class Variable:
    def __init__(self, data: np.ndarray) -> None:
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError("{} is not supported".format(type(data)))

        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            x, y = f.input, f.output
            x.grad = f.backward(y.grad)

            if x.creator is not None:
                funcs.append(x.creator)


# %%
def as_array(x: Union[np.ndarray, np.generic]) -> np.ndarray:
    if np.isscalar(x):
        return np.array(x)
    return x


# %%
class Function:
    def __call__(self, inputs: Iterable[Variable]) -> list[Variable]:
        xs = [x.data for x in inputs]  # Get data from Variable
        ys = self.forward(xs)
        outputs = [Variable(as_array(y)) for y in ys]  # Wrap data

        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = outputs
        return outputs

    def forward(self, xs: Iterable[np.ndarray]) -> tuple[np.ndarray]:
        raise NotImplementedError()

    def backward(self, gys: Iterable[np.ndarray]) -> tuple[np.ndarray]:
        raise NotImplementedError()


# %%
class Add(Function):
    def forward(self, xs: Iterable[np.ndarray]) -> tuple[np.ndarray]:
        x0, x1 = xs
        y = x0 + x1
        return (y,)


# %%
xs = [Variable(np.array(2)), Variable(np.array(3))]
f = Add()
ys = f(xs)
y = ys[0]
print(y.data)
