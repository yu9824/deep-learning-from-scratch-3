# %% [markdown]
# `.backward` で np.ndarrayとして演算してたのを `Variable` として演算させるように修正する。 `Variable` クラスは、自動で計算グラフを作るように設計されている。
