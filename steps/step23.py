# %%
# Add import path for the dezero directory.
import os
import sys
from pathlib import Path

if "__file__" in globals():
    sys.path.append(str((Path(__file__).parent / "..").resolve()))
else:
    sys.path.append(str(Path(os.getcwd(), "..").resolve()))


# %%
import numpy as np

from dezero.core_simple import Variable


# %%
x = Variable(np.array(1.0))
y = (x + 3) ** 2
y.backward()

# %%
print(y)
print(x.grad)
