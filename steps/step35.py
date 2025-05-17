# %%
import os
import sys
from pathlib import Path

import numpy as np

if "__file__" in globals():
    sys.path.append(str((Path(__file__).parent / "..").resolve()))
else:
    sys.path.append(str(Path(os.getcwd(), "..").resolve()))
import dezero.functions as F
from dezero import Variable
from dezero.utils import plot_dot_graph

# %%
x = Variable(np.array(1.0))
y = F.tanh(x)
x.name = "x"
y.name = "y"
y.backward(create_graph=True)

# %%
iters = 1

# %%
for i in range(iters):
    gx = x.grad
    x.cleargrad()
    gx.backward(create_graph=True)

# %%
gx = x.grad
gx.name = "gx" + str(iters + 1)
plot_dot_graph(gx, verbose=False, to_file="tanh.png")
