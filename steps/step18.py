# %% [markdown]
# メモリ使用量を減らす
#
# 1. 不要な微分を消去する仕組みを導入する
# 2. 「逆伝搬が必要ない場合のモード」を用意する

# %%
import contextlib
import weakref

import numpy as np


# %%
class Config:
    enable_backprop = True
    """
    - True: 学習時 (微分計算のため逆伝搬させる)
    - False: 推論時 (逆伝搬しない → メモリを節約)
    """


# %%
@contextlib.contextmanager
def using_config(name, value):
    """with文を使ってconfigを一時的に変更する

    Parameters
    ----------
    name : str
        attribute name
    value : Any
        value
    """
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)


# %%
def no_grad():
    return using_config("enable_backprop", False)


# %%
class Variable:
    def __init__(self, data: np.ndarray) -> None:
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError("{} is not supported".format(type(data)))

        self.data = data
        self.grad = None
        self.creator = None
        self.generation = 0

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def cleargrad(self):
        self.grad = None

    def backward(self, retain_grad=False):
        """

        Parameters
        ----------
        retain_grad : bool, optional
            勾配を保持するかどうか, by default False
        """
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)

        while funcs:
            f = funcs.pop()
            gys = [output().grad for output in f.outputs]  # output is weakref
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
                    # 勾配を消去
                    y().grad = None  # y is weakref


# %%
def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


# %%
class Function:
    def __call__(self, *inputs: Variable):
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        if Config.enable_backprop:
            # self.inputsが必要なのは微分を計算 (逆伝搬) するときだけ。
            # 逆伝搬しないときには保持させないことでメモリを節約する
            # self.generationとself.inputs
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)
            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]

        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, xs):
        raise NotImplementedError()

    def backward(self, gys):
        raise NotImplementedError()


# %%
class Square(Function):
    def forward(self, x):
        y = x**2
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx


# %%
def square(x: Variable) -> Variable:
    return Square()(x)


# %%
class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y

    def backward(self, gy):
        return gy, gy


# %%
def add(x0: Variable, x1: Variable) -> Variable:
    return Add()(x0, x1)


# %%
x0 = Variable(np.array(1.0))
x1 = Variable(np.array(1.0))
t = add(x0, x1)
y = add(x0, t)
y.backward()
print(y.grad, t.grad)  # None None
print(x0.grad, x1.grad)  # 2.0 1.0


# %%
with using_config("enable_backprop", False):
    x = Variable(np.array(2.0))
    y = square(x)

# %%
with no_grad():
    x = Variable(np.array(2.0))
    y = square(x)

# %%
x = Variable(np.array(2.0))
y = square(x)
y.backward()
x.grad

# %%
with no_grad():
    x = Variable(np.array(2.0))
    y = square(x)

    # 逆伝搬用の計算をしていないのでエラーになる
    y.backward()
