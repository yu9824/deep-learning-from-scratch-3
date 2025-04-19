# %% [markdown]
# `Variable` を `np.ndarray` やスカラーと演算できるようにする

# %%
import weakref
import numpy as np
import contextlib


# %%
class Config:
    enable_backprop = True


# %%
@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)


# %%
def no_grad():
    return using_config("enable_backprop", False)


# %% [markdown]
# ### 二項演算子の実装について
#
# 二項演算子における呼び出される順番の話。足し算などの四則演算など。
#
# 以下の場合にどのような順番でメソッドが呼ばれるのか。
#
# ```python
# a + b
# ```
#
# 基本的には、
#
# 1. `a.__add__` が呼ばれる
# 2. (1. で `NotImplemented` が返ってきたときには) `b.__radd__` が呼ばれる
#
# たとえば、以下の例では、 1 (int) の `__add__` が呼ばれるが、 `np.ndarray` との演算は定義されておらず、 `NotImplemented` が帰ってくる。なので、 `np.ndarray.__radd__` が呼ばれる。結果演算できる。
#
# ```python
# 1 + np.array(1.0)
# ```

# %% [markdown]
# ### \_\_array_priority\_\_について
#
# 今回の `Variable` のように `np.ndarray` との演算をしたい場合、に困る。
# `np.ndarray.__add__` が強すぎる。 全然 `NotImplemented` にならないため、`np.ndarray` にされて演算されてしまう。
#
# なので、それもoverwrapしたい場合は、 `__array_priority__` というattributeを設定して、ある程度大きな値にする ( `np.ndarray.__array_priority__` のデフォルトは0 )。

# %%
class MyArray:
    __array_priority__ = 0

    def __array__(self, dtype=None):
        return np.array([10, 20, 30])

    def __radd__(self, other):
        print("MyArray.__radd__ called")
        return "My custom add result"


# 優先度の設定（デフォルトは0）
a = np.array([1, 2, 3])
b = MyArray()

print(f"{a + b}")

# 優先度を高く設定
MyArray.__array_priority__ = 1000

print(f"{a + b}")


# %%
class Variable:
    # np.ndarray との演算でこっちの演算子を優先させる
    __array_priority__ = 200

    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError("{} is not supported".format(type(data)))

        self.data = data
        self.name = name
        self.grad = None
        self.creator = None
        self.generation = 0

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

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def cleargrad(self):
        self.grad = None

    def backward(self, retain_grad=False):
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
                    y().grad = None  # y is weakref


# %%
def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)


# %%
def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


# %%
class Function:
    def __call__(self, *inputs):
        # as_variableをつかうことで全部 `Variable` に揃える
        inputs = [as_variable(x) for x in inputs]

        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        if Config.enable_backprop:
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
class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y

    def backward(self, gy):
        return gy, gy


# %%
def add(x0, x1):
    x1 = as_array(x1)
    return Add()(x0, x1)


# %%
class Mul(Function):
    def forward(self, x0, x1):
        y = x0 * x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return gy * x1, gy * x0


# %%
def mul(x0, x1):
    x1 = as_array(x1)
    return Mul()(x0, x1)


# %%
type((3.0).__mul__(np.array(3.0)))

# %%
Variable.__add__ = add
Variable.__radd__ = add
Variable.__mul__ = mul
Variable.__rmul__ = mul

# %%
x = Variable(np.array(2.0))
y = x + np.array(3.0)
print(y)

# %%
y = x + 3.0
print(y)

# %%
y = 3.0 * x + 1.0
print(y)
