# %%
from typing import Union

import numpy as np


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

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = []
        # 分岐で同じ親を登録しないようにするため。
        # 独自で定義したclassから生成したinstanceは、hashable ← 知らなかった。
        # 非明示的にobjectを継承しており、object.__hash__が定義されているから。
        # hashableなのでsetで重複管理できる。
        seen_set = set()

        def add_func(f):
            # idを確認すべくprint
            print(f)
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                # 世代でソート (世代が一番大きいやつを取り出したいだけなので、必ずしもソートする必要はない)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)

        while funcs:
            f = funcs.pop()
            gys = [output.grad for output in f.outputs]
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


# %%
def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


# %%
class Function:
    def __call__(
        self, *inputs: Variable
    ) -> Union[tuple[Variable, ...], Variable]:
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = tuple(Variable(as_array(y)) for y in ys)

        self.generation = max([x.generation for x in inputs])
        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = outputs
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
x = Variable(np.array(2.0))
a = square(x)
y = add(square(a), square(a))
y.backward()

# %%
print(y.data)
print(x.grad)
