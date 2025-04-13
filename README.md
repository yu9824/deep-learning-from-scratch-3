```bash
# 演習用環境構築
conda create -yn dlfs3 python=3.12 graphviz cupy matplotlib jupyter jupytext jupyter-book numpy pillow

```

> <p><a href="https://www.amazon.co.jp/dp/4873119065/ref=cm_sw_r_tw_dp_U_x_KiA1Eb39SW14Q"><img src="https://raw.githubusercontent.com/oreilly-japan/deep-learning-from-scratch-3/images/deep-learning-from-scratch-3.png" height="250" /></a></p>
>
> ## 本書概要
>
> 本書では「DeZero」というディープラーニングのフレームワークを作ります。DeZeroは本書オリジナルのフレームワークです。最小限のコードで、フレームワークのモダンな機能を実現します。本書では、この小さな——それでいて十分にパワフルな——フレームワークを、全部で60のステップで完成させます。それによって、PyTorch、TensorFlow、Chainerなどの現代のフレームワークに通じる深い知識を養います。
>
>
> <p><img src="https://raw.githubusercontent.com/oreilly-japan/deep-learning-from-scratch-3/images/dezero_logo.png" width="400px" /></p>
>
>
> <p><a href="https://pypi.python.org/pypi/dezero"><img alt="pypi" src="https://img.shields.io/pypi/v/dezero.svg" /></a><a href="https://github.com/oreilly-japan/deep-learning-from-scratch-3/blob/master/LICENSE.md"><img alt="MIT License" src="http://img.shields.io/badge/license-MIT-blue.svg" /></a><a href="https://travis-ci.org/oreilly-japan/deep-learning-from-scratch-3"><img alt="Build Status" src="https://travis-ci.org/oreilly-japan/deep-learning-from-scratch-3.svg?branch=master" /></a></p>
>
> ## ニュース
>
> <a href="https://koki0702.github.io/dezero-book/"><img src="https://raw.githubusercontent.com/koki0702/koki0702.github.io/master/dezero-book/images/summary_ja.png" height="150px" /></a>
>
> 【試し読み】本書の一部をオンラインで公開しています。
> https://koki0702.github.io/dezero-book/
>
>
> ## ファイル構成
>
> | フォルダ名            | 説明                                    |
> | :-------------------- | :-------------------------------------- |
> | [dezero](/dezero)     | DeZeroのソースコード                    |
> | [examples](/examples) | DeZeroを使った実装例                    |
> | [steps](/steps)       | 各stepファイル（step01.py ~ step60.py） |
> | [tests](/tests)       | DeZeroのユニットテスト                  |
>
>
> ## 必要な外部ライブラリ
>
> 本書で使用するPytnonのバージョンと外部ライブラリは下記の通りです。
>
> - [Python 3系](https://docs.python.org/3/)
> - [NumPy](https://numpy.org/)
> - [Matplotlib](https://matplotlib.org/)
>
> またオプションとして、NVIDIAのGPUで実行できる機能も提供します。その場合は下記のライブラリが必要です。
>
> - [CuPy](https://cupy.chainer.org/) （オプション）
>
>
> ## 実行方法
>
> 本書で説明するPythonファイルは、主に[steps](/steps)ファルダにあります。
> 実行するためには、下記のとおりPythonコマンドを実行します（どのディレクトリからでも実行できます）。
>
> ```bash
> $ python steps/step01.py
> $ python steps/step02.py
>
> $ cd steps
> $ python step31.py
> ```
>
> ## デモ
>
> DeZeroの他の実装例は[examples](/examples)にあります。
>
> [<img src="https://raw.githubusercontent.com/oreilly-japan/deep-learning-from-scratch-3/images/example_tanh.png" height="175"/>](https://github.com/oreilly-japan/deep-learning-from-scratch-3/tree/tanh)[<img src="https://raw.githubusercontent.com/oreilly-japan/deep-learning-from-scratch-3/images/example_spiral.png" height="175"/>](/examples/spiral.py)[<img src="https://raw.githubusercontent.com/oreilly-japan/deep-learning-from-scratch-3/images/example_gpu.png" height="175"/>](https://colab.research.google.com/github/oreilly-japan/deep-learning-from-scratch-3/blob/master/examples/mnist_colab_gpu.ipynb)
>
> [<img src="https://raw.githubusercontent.com/oreilly-japan/deep-learning-from-scratch-3/images/gan.gif" height="175"/>](/examples/gan.py)[<img src="https://raw.githubusercontent.com/oreilly-japan/deep-learning-from-scratch-3/images/vae.png" height="175"/>](/examples/vae.py)[<img src="https://raw.githubusercontent.com/oreilly-japan/deep-learning-from-scratch-3/images/grad_cam.png" height="175"/>](/examples/grad_cam.py)
>
> [<img src="https://raw.githubusercontent.com/oreilly-japan/deep-learning-from-scratch-3/images/style_transfer.png" height="175"/>](/examples/style_transfer.py)[<img src="https://raw.githubusercontent.com/oreilly-japan/deep-learning-from-scratch-3/images/pythonista.png" height="175"/>](https://github.com/oreilly-japan/deep-learning-from-scratch-3/wiki/DeZero%E3%82%92iPhone%E3%81%A7%E5%8B%95%E3%81%8B%E3%81%99)
>
> ## 正誤表
>
> 本書の正誤情報は、[:mag_right: 正誤表ページ](../../wiki/Errata)に掲載しています。
>
> 正誤表ページに掲載されていない誤植や間違いなどを見つけた方は、[:email: japan@oreilly.co.jp](<mailto:japan@oreilly.co.jp>)までお知らせください。
>
