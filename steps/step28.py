# %%
import os
import sys
from array import array
from pathlib import Path
from typing import TypeVar

if "__file__" in globals():
    sys.path.append(str((Path(__file__).parent / "..").resolve()))
else:
    sys.path.append(str(Path(os.getcwd(), "..").resolve()))
import numpy as np

# import dezero's simple_core explicitly
# import dezero
from dezero.core_simple import Variable

# if not dezero.is_simple_core:
#     from dezero.core_simple import Variable, setup_variable

#     setup_variable()

# %%
T = TypeVar("T", Variable, int, float, np.ndarray, np.generic)


# %%
def rosenbrock(x0: T, x1: T) -> T:
    y = 100 * (x1 - x0**2) ** 2 + (x0 - 1) ** 2
    return y


# %%
x0 = Variable(np.array(0.0))
x1 = Variable(np.array(2.0))
lr = 0.001
iters = 1000

# %%
y = rosenbrock(x0, x1)
y.backward()

print(x0.grad, x1.grad)

# %%
arr_x0 = array("f", [x0.data.item()])
arr_x1 = array("f", [x1.data.item()])

for i in range(iters):
    print(x0, x1)

    y = rosenbrock(x0, x1)

    x0.cleargrad()
    x1.cleargrad()
    y.backward()

    x0.data -= lr * x0.grad
    x1.data -= lr * x1.grad

    arr_x0.append(x0.data.item())
    arr_x1.append(x1.data.item())

# %%
import matplotlib.colors
import matplotlib.pyplot as plt


# %%
fig, ax = plt.subplots()

_x_for_plot = np.linspace(-2, 2, 1001)
_y_for_plot = np.linspace(-1, 3, 1001)
_z_for_plot = rosenbrock(*np.meshgrid(_x_for_plot, _y_for_plot))

# 最小値が0のため、対数にするとエラーになる。0に最小値の1/2を代入する
print(f"{_z_for_plot.min()=}")
print(f"{_z_for_plot.max()=}")
_z_for_plot[np.isclose(_z_for_plot, 0.0)] = (
    _z_for_plot[_z_for_plot > 0].min() / 2
)

kwargs_contour = dict(
    cmap="viridis",
    norm=matplotlib.colors.LogNorm(),
)

ax.contour(
    _x_for_plot,
    _y_for_plot,
    _z_for_plot,
    **kwargs_contour,
)

# 最小点をプロット
_idx_y_min, _idx_x_min = np.unravel_index(
    np.argmin(_z_for_plot), shape=_z_for_plot.shape
)
ax.scatter(
    _x_for_plot[_idx_x_min],
    _y_for_plot[_idx_y_min],
    marker="*",
    color="black",
    s=150,
    zorder=5.0,
    label="minimum",
)

# 軌跡のプロット
ax.plot(arr_x0, arr_x1, marker="o", color="red", label="traj")

# ダミーのフレームにcontourfをプロット。
# これを使用してカラーバーを書く
hidden_ax = fig.add_subplot(111, frame_on=False)
hidden_ax.set_visible(False)
mappable = hidden_ax.contourf(
    _x_for_plot,
    _y_for_plot,
    _z_for_plot,
    **kwargs_contour,
)

ax.set_xlabel("x0")
ax.set_ylabel("x1")
ax.legend()

fig.colorbar(mappable, ax=ax)
fig.tight_layout()
