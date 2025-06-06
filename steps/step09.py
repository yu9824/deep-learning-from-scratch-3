# %% [markdown]
# 使いやすくする

# %%
from typing import Optional
import numpy as np


# %%
class Variable:
    def __init__(self, data: Optional[np.ndarray]) -> None:
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
            self.grad = np.ones_like(
                self.data
            )  # データ型を揃えるため。（複数の変数を利用することを想定しているという観点もあるのでは？）

        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            x, y = f.input, f.output
            x.grad = f.backward(y.grad)

            if x.creator is not None:
                funcs.append(x.creator)


# %%
def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


# %%
class Function:
    def __call__(self, input: Variable) -> Variable:
        x = input.data
        y = self.forward(x)
        # 処理によってはスカラーとして帰ってくる。ndarrayで統一する。
        output = Variable(as_array(y))
        output.set_creator(self)
        self.input = input
        self.output = output
        return output

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()


# %%
class Square(Function):
    def forward(self, x):
        y = x**2
        return y

    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx


# %%
class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y

    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx


# %%
# overwrap for easy use
def square(x):
    return Square()(x)


# %%
# overwrap for easy use
def exp(x):
    return Exp()(x)


# %%
x = Variable(np.array(0.5))
y = square(exp(square(x)))
y.backward()
print(x.grad)


# %%
x = Variable(np.array(1.0))  # OK
x = Variable(None)  # OK
x = Variable(1.0)  # NG
