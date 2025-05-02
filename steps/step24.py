# %% [markdown]
# [Test functions for optimization](https://en.wikipedia.org/wiki/Test_functions_for_optimization)で紹介されている最適化手法のベンチマーク用の関数を使って、いくつか微分を試してみる。

# %%
# Add import path for the dezero directory.
import os
import sys
from pathlib import Path

if "__file__" in globals():
    sys.path.append(str((Path(__file__).parent / "..").resolve()))
else:
    sys.path.append(str(Path(os.getcwd(), "..").resolve()))

import numpy as np

from dezero.core_simple import Variable


# %%
def sphere(x: Variable, y: Variable) -> Variable:
    z = x**2 + y**2
    return z


# %%
def matyas(x: Variable, y: Variable) -> Variable:
    z = 0.26 * (x**2 + y**2) - 0.48 * x * y
    return z


# %%
def goldstein(x: Variable, y: Variable) -> Variable:
    z = (
        1
        + (x + y + 1) ** 2
        * (19 - 14 * x + 3 * x**2 - 14 * y + 6 * x * y + 3 * y**2)
    ) * (
        30
        + (2 * x - 3 * y) ** 2
        * (18 - 32 * x + 12 * x**2 + 48 * y - 36 * x * y + 27 * y**2)
    )
    return z


# %%
x = Variable(np.array(1.0))
y = Variable(np.array(1.0))

# %%
z = sphere(x, y)
z.backward()
print(x.grad, y.grad)

# %%
# 勾配初期化
x.cleargrad()
y.cleargrad()

# %%
z = matyas(x, y)
z.backward()
print(x.grad, y.grad)

# %%
# 勾配初期化
x.cleargrad()
y.cleargrad()

# %%
z = goldstein(x, y)  # sphere(x, y) / matyas(x, y)
z.backward()
print(x.grad, y.grad)
