# %%
import os
import sys
from pathlib import Path

import numpy as np

if "__file__" in globals():
    sys.path.append(str((Path(__file__).parent / "..").resolve()))
else:
    sys.path.append(str(Path(os.getcwd(), "..").resolve()))
from dezero import Variable

# %%
def f(x):
    y = x**4 - 2 * x**2
    return y

# %%
x = Variable(np.array(2.0))
iters = 10

# %%
for i in range(iters):
    print(i, x)

    y = f(x)
    x.cleargrad()
    y.backward(create_graph=True)

    gx = x.grad
    x.cleargrad()
    gx.backward()
    gx2 = x.grad

    x.data -= gx.data / gx2.data
