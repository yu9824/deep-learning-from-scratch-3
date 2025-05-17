import abc
import contextlib
import importlib.util
import weakref
from typing import Any, Optional, Union

import numpy as np

import dezero


# =============================================================================
# Config
# =============================================================================
class Config:
    enable_backprop = True
    train = True


@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)


def no_grad():
    return using_config("enable_backprop", False)


def test_mode():
    return using_config("train", False)


# =============================================================================
# Variable / Function
# =============================================================================
if importlib.util.find_spec("cupy") is not None:
    import cupy

    array_types = (np.ndarray, cupy.ndarray)
    _ArrayType = Union[np.ndarray, cupy.ndarray]
else:
    array_types = (np.ndarray,)  # type: ignore[assignment]
    _ArrayType = np.ndarray  # type: ignore[misc]

# To escape 'reportInvalidTypeForm'
ArrayType = _ArrayType


class Variable:
    __array_priority__ = 200

    def __init__(self, data: Optional[ArrayType], name=None):
        if data is not None:
            if not isinstance(data, array_types):
                raise TypeError("{} is not supported".format(type(data)))

        self.data = data
        self.name = name
        self.grad: Optional[Variable] = None
        self.creator: Optional["Function"] = None
        self.generation: int = 0

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if self.data is None:
            return "variable(None)"
        p = str(self.data).replace("\n", "\n" + " " * 9)
        return "variable(" + p + ")"

    def set_creator(self, func: "Function"):
        self.creator = func
        self.generation = func.generation + 1

    def unchain(self):
        self.creator = None

    def cleargrad(self):
        self.grad = None

    def backward(self, retain_grad=False, create_graph=False):
        if self.grad is None:
            xp = dezero.cuda.get_array_module(self.data)
            self.grad = Variable(xp.ones_like(self.data))

        funcs: list[Function] = []  # type: ignore[annotation-unchecked]
        seen_set: set[Function] = set()  # type: ignore[annotation-unchecked]

        def add_func(f: "Function"):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)
        while funcs:
            f = funcs.pop()
            gys = [output().grad for output in f.outputs]  # output is weakref

            with using_config("enable_backprop", create_graph):
                gxs = f.backward(*gys)
                if not isinstance(gxs, tuple):
                    gxs = (gxs,)

                for x, gx in zip(f.inputs, gxs):
                    if x.grad is None:
                        x.grad = gx
                    else:
                        x.grad = x.grad + gx

                    if x.creator is not None:
                        add_func(x.creator)

            if not retain_grad:
                for y in f.outputs:
                    y().grad = None  # y is weakref

    def unchain_backward(self):
        if self.creator is not None:
            funcs = [self.creator]
            while funcs:
                f = funcs.pop()
                for x in f.inputs:
                    if x.creator is not None:
                        funcs.append(x.creator)
                        x.unchain()

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return dezero.functions.reshape(self, shape)

    def transpose(self, *axes):
        if len(axes) == 0:
            axes = None
        elif len(axes) == 1:
            if isinstance(axes[0], (tuple, list)) or axes[0] is None:
                axes = axes[0]
        return dezero.functions.transpose(self, axes)

    @property
    def T(self):
        return dezero.functions.transpose(self)

    def sum(self, axis=None, keepdims=False):
        return dezero.functions.sum(self, axis, keepdims)

    def to_cpu(self):
        if self.data is not None:
            self.data = dezero.cuda.as_numpy(self.data)

    def to_gpu(self):
        if self.data is not None:
            self.data = dezero.cuda.as_cupy(self.data)

    def __add__(self, *args) -> "Variable":
        return add(self, *args)

    def __radd__(self, *args) -> "Variable":
        return add(self, *args)

    def __mul__(self, *args) -> "Variable":
        return mul(self, *args)

    def __rmul__(self, *args) -> "Variable":
        return mul(self, *args)

    def __neg__(self, *args) -> "Variable":
        return neg(self, *args)

    def __sub__(self, *args) -> "Variable":
        return sub(self, *args)

    def __rsub__(self, *args) -> "Variable":
        return rsub(self, *args)

    def __truediv__(self, *args) -> "Variable":
        return div(self, *args)

    def __rtruediv__(self, *args) -> "Variable":
        return rdiv(self, *args)

    def __pow__(self, *args) -> "Variable":
        return pow(self, *args)

    def __getitem__(self, *args):
        return dezero.functions.get_item(self, *args)

    def matmul(self, *args) -> "Variable":
        return dezero.functions.matmul(self, *args)

    def dot(self, *args) -> "Variable":
        return dezero.functions.matmul(self, *args)

    def max(self, *args) -> "Variable":
        return dezero.functions.max(self, *args)

    def min(self, *args) -> "Variable":
        return dezero.functions.min(self, *args)


class Parameter(Variable):
    pass


def as_variable(obj: Any) -> Variable:
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)


def as_array(x: Any, array_module=np) -> ArrayType:
    if np.isscalar(x):
        return array_module.array(x)
    return x


class Function(abc.ABC):
    def __call__(
        self, *inputs: Union[Variable, ArrayType]
    ) -> Union[
        Variable,
        list[Variable],
    ]:
        inputs = tuple(as_variable(x) for x in inputs)

        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        if Config.enable_backprop:
            self.generation = max(x.generation for x in inputs)  # type: ignore[union-attr]
            for output in outputs:
                output.set_creator(self)
            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]

        return outputs if len(outputs) > 1 else outputs[0]

    @abc.abstractmethod
    def forward(
        self, *xs: ArrayType
    ) -> Union[
        ArrayType,
        tuple[ArrayType, ...],
    ]:
        raise NotImplementedError()

    @abc.abstractmethod
    def backward(
        self, gys: Union[Variable, ArrayType]
    ) -> Union[
        ArrayType,
        tuple[ArrayType, ...],
    ]:
        raise NotImplementedError()


# =============================================================================
# 四則演算 / 演算子のオーバーロード
# =============================================================================
class Add(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 + x1
        return y

    def backward(self, gy):
        gx0, gx1 = gy, gy
        if self.x0_shape != self.x1_shape:  # for broadcaset
            gx0 = dezero.functions.sum_to(gx0, self.x0_shape)
            gx1 = dezero.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1


def add(x0, x1):
    x1 = as_array(x1, dezero.cuda.get_array_module(x0.data))
    return Add()(x0, x1)


class Mul(Function):
    def forward(self, x0, x1):
        y = x0 * x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs
        gx0 = gy * x1
        gx1 = gy * x0
        if x0.shape != x1.shape:  # for broadcast
            gx0 = dezero.functions.sum_to(gx0, x0.shape)
            gx1 = dezero.functions.sum_to(gx1, x1.shape)
        return gx0, gx1


def mul(x0, x1):
    x1 = as_array(x1, dezero.cuda.get_array_module(x0.data))
    return Mul()(x0, x1)


class Neg(Function):
    def forward(self, x):
        return -x

    def backward(self, gy):
        return -gy


def neg(x):
    return Neg()(x)


class Sub(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 - x1
        return y

    def backward(self, gy):
        gx0 = gy
        gx1 = -gy
        if self.x0_shape != self.x1_shape:  # for broadcast
            gx0 = dezero.functions.sum_to(gx0, self.x0_shape)
            gx1 = dezero.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1


def sub(x0, x1):
    x1 = as_array(x1, dezero.cuda.get_array_module(x0.data))
    return Sub()(x0, x1)


def rsub(x0, x1):
    x1 = as_array(x1, dezero.cuda.get_array_module(x0.data))
    return Sub()(x1, x0)


class Div(Function):
    def forward(self, x0, x1):
        y = x0 / x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1**2)
        if x0.shape != x1.shape:  # for broadcast
            gx0 = dezero.functions.sum_to(gx0, x0.shape)
            gx1 = dezero.functions.sum_to(gx1, x1.shape)
        return gx0, gx1


def div(x0, x1):
    x1 = as_array(x1, dezero.cuda.get_array_module(x0.data))
    return Div()(x0, x1)


def rdiv(x0, x1):
    x1 = as_array(x1, dezero.cuda.get_array_module(x0.data))
    return Div()(x1, x0)


class Pow(Function):
    def __init__(self, c):
        self.c = c

    def forward(self, x):
        y = x**self.c
        return y

    def backward(self, gy):
        (x,) = self.inputs
        c = self.c
        gx = c * x ** (c - 1) * gy
        return gx


def pow(x, c):
    return Pow(c)(x)


def setup_variable():
    pass
