from __future__ import annotations

import abc
import contextlib
import weakref
from abc import abstractmethod
from typing import Any, Optional, TypeVar, Union, cast, overload

import numpy as np


# =============================================================================
# Config
# =============================================================================
class Config:
    enable_backprop = True


@contextlib.contextmanager
def using_config(name: str, value: Any):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)


def no_grad():
    return using_config("enable_backprop", False)


# =============================================================================
# Variable / Function
# =============================================================================
class Variable:
    __array_priority__: float = 200.0

    def __add__(self, *args):
        return add(self, *args)

    def __radd__(self, *args):
        return add(self, *args)

    def __mul__(self, *args):
        return mul(self, *args)

    def __rmul__(self, *args):
        return mul(self, *args)

    def __neg__(self, *args):
        return neg(self, *args)

    def __sub__(self, *args):
        return sub(self, *args)

    def __rsub__(self, *args):
        return rsub(self, *args)

    def __truediv__(self, *args):
        return div(self, *args)

    def __rtruediv__(self, *args):
        return rdiv(self, *args)

    def __pow__(self, *args):
        return pow(self, *args)

    def __init__(
        self, data: Optional[np.ndarray], name: Optional[str] = None
    ) -> None:
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError("{} is not supported".format(type(data)))

        self.data = data
        self.name = name
        self.grad: Optional[Union[np.generic, np.ndarray]] = None
        self.creator: Optional[Function] = None
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

    def set_creator(self, func: Function):
        self.creator = func
        self.generation = func.generation + 1

    def cleargrad(self):
        self.grad = None

    def backward(self, retain_grad: bool = False):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs: list[Function] = []
        seen_set: set[Function] = set()

        def add_func(f: Function):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        assert isinstance(self.creator, Function)
        add_func(self.creator)

        while funcs:
            f = funcs.pop()

            # weakrefが切れたときにNoneが返ってくる。
            # その型チェックは困難なので今回は諦める
            gys = [
                output().grad  # type: ignore[union-attr]
                for output in f.outputs
            ]  # output is weakref

            gxs = f.backward(*gys)  # type: ignore[arg-type]
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx  # type: ignore[operator]

                if x.creator is not None:
                    add_func(x.creator)

            if not retain_grad:
                for y in f.outputs:
                    # weakrefが切れたときにNoneが返ってくる。
                    # その型チェックは困難なので今回は諦める
                    y().grad = None  # type: ignore[union-attr] # y is weakref


def as_variable(obj: Any) -> Variable:
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)


ArrayOrVar = TypeVar("ArrayOrVar", np.ndarray, Variable)


@overload
def as_array(x: Union[np.generic, int, float]) -> np.ndarray: ...


@overload
def as_array(x: ArrayOrVar) -> ArrayOrVar: ...


# HACK: 3.10以上では、TypeGuardを使った方がよりかっこよく定義できそう
# そもそもこの関数がちょっと微妙な気がする。Variableに拡張した時点で何かをした方が良かった気がする。
def as_array(
    x: Union[np.generic, int, float, ArrayOrVar],
) -> Union[np.ndarray, ArrayOrVar]:
    if np.isscalar(x):
        return np.array(x)
    else:
        assert isinstance(x, (np.ndarray, Variable))
        return x


class Function(abc.ABC):
    def __call__(
        self, *inputs: Union[Variable, np.ndarray, np.generic]
    ) -> Union[
        Variable,
        list[Variable],
    ]:
        inputs = tuple(as_variable(x) for x in inputs)
        # for type check
        inputs = cast(tuple[Variable, ...], inputs)

        xs = [x.data for x in inputs if isinstance(x.data, np.ndarray)]
        assert len(xs) == len(inputs)  # x.data is Noneのデータがないことを確認
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        assert isinstance(ys, tuple)
        outputs = [Variable(as_array(y)) for y in ys]

        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)
            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]

        return outputs if len(outputs) > 1 else outputs[0]

    @abstractmethod
    def forward(
        self, *xs: Union[np.ndarray, np.generic]
    ) -> Union[
        Union[np.ndarray, np.generic],
        tuple[Union[np.ndarray, np.generic], ...],
    ]:
        raise NotImplementedError()

    @abstractmethod
    def backward(
        self, gys: Union[Variable, np.ndarray, np.generic]
    ) -> Union[
        Union[np.ndarray, np.generic],
        tuple[Union[np.ndarray, np.generic], ...],
    ]:
        raise NotImplementedError()


# =============================================================================
# 四則演算 / 演算子のオーバーロード
# =============================================================================
class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y

    def backward(self, gy):
        return gy, gy


def add(x0, x1):
    x1 = as_array(x1)
    out = Add()(x0, x1)
    assert isinstance(out, Variable)
    return out


class Mul(Function):
    def forward(self, x0, x1):
        y = x0 * x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return gy * x1, gy * x0


def mul(x0, x1):
    x1 = as_array(x1)
    out = Mul()(x0, x1)
    assert isinstance(out, Variable)
    return out


class Neg(Function):
    def forward(self, x):
        return -x

    def backward(self, gy):
        return -gy


def neg(x):
    out = Neg()(x)
    assert isinstance(out, Variable)
    return out


class Sub(Function):
    def forward(self, x0, x1):
        y = x0 - x1
        return y

    def backward(self, gy):
        return gy, -gy


def sub(x0, x1):
    x1 = as_array(x1)
    out = Sub()(x0, x1)
    assert isinstance(out, Variable)
    return out


def rsub(x0, x1):
    x1 = as_array(x1)
    out = Sub()(x1, x0)
    assert isinstance(out, Variable)
    return out


class Div(Function):
    def forward(self, x0, x1):
        y = x0 / x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1**2)
        return gx0, gx1


def div(x0, x1):
    x1 = as_array(x1)
    out = Div()(x0, x1)
    assert isinstance(out, Variable)
    return out


def rdiv(x0, x1):
    x1 = as_array(x1)
    out = Div()(x1, x0)
    assert isinstance(out, Variable)
    return out


class Pow(Function):
    def __init__(self, c):
        self.c = c

    def forward(self, x):
        y = x**self.c
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        c = self.c

        gx = c * x ** (c - 1) * gy
        return gx


def pow(x, c):
    out = Pow(c)(x)
    assert isinstance(out, Variable)
    return out


def setup_variable():
    Variable.__add__ = add
    Variable.__radd__ = add
    Variable.__mul__ = mul
    Variable.__rmul__ = mul
    Variable.__neg__ = neg
    Variable.__sub__ = sub
    Variable.__rsub__ = rsub
    Variable.__truediv__ = div
    Variable.__rtruediv__ = rdiv
    Variable.__pow__ = pow
