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
x = Variable(np.array(2.0))
y = x**2
y.backward(create_graph=True)
gx = x.grad
x.cleargrad()

# %%
z = gx**3 + y
z.backward()
print(x.grad)
