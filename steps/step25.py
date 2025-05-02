# %% [markdown]
# DOT言語を使用してグラフを書いてみる
#
# grapyvizの `dot` コマンドを使用して書く。

# %%
from IPython.display import Image

# %%
sample_dot = """
digraph g{
x
y
}
"""

# %%
with open("./sample.dot", mode="w", encoding="utf-8") as f:
    f.write(sample_dot)

# %%
# !dot sample.dot -T png -o sample.png

# %%
Image(filename="./sample.png")

# %%
sample_dot = """
digraph g{
1 [label="x", color=orange, style=filled]           # nodeの定義 []内でpropertyを定義
2 [label="y", color=orange, style=filled]
3 [label="Exp", color=lightblue, style=filled, shape=box]
1 -> 3  # edgeの定義
3 -> 2
}
"""

# %%
with open("./sample.dot", mode="w", encoding="utf-8") as f:
    f.write(sample_dot)

# %%
# !dot sample.dot -T png -o sample.png

# %%
Image(filename="./sample.png")
