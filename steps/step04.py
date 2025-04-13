# %%
from collections.abc import Callable

import numpy as np


# %%
class Variable:
    def __init__(self, data: np.ndarray) -> None:
        self.data = data


# %%
class Function:
    def __call__(self, input: Variable) -> Variable:
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        self.input = input
        self.output = output
        return output

    def forward(self, x):
        raise NotImplementedError()


# %%
class Square(Function):
    def forward(self, x):
        return x**2


# %%
class Exp(Function):
    def forward(self, x):
        return np.exp(x)


# %%
def numerical_diff(
    f: Callable[[Variable], Variable], x: Variable, eps: float = 1e-4
):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)


# %%
f = Square()
x = Variable(np.array(2.0))
dy = numerical_diff(f, x)
print(dy)


# %% [markdown]
# 合成関数の微分


# %%
def f(x: Variable) -> Variable:
    A = Square()
    B = Exp()
    C = Square()
    return C(B(A(x)))


# %%
x = Variable(np.array(0.5))
dy = numerical_diff(f, x)
print(dy)

# %% [markdown]
# 数値微分は桁落ちによって誤差が大きくなる場合あり。
#
# → バックプロパゲーション・勾配確認
