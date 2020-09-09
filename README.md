# ゼロから作る Deep Learning

## 概要

『ゼロから作る Deep Learning』を進めていく

## URL

- [オライリージャパン](https://www.oreilly.co.jp/books/9784873117584/)
- [GitHub（オライリージャパン）](https://github.com/oreilly-japan/deep-learning-from-scratch)

## 注意点

### 4 章

train_neuralnet.py（p.120）は「1 エポックあたりの繰り返し数」の位置が違う。また、ハイパーパラメータの train_size も消してはいけない。

``` python
...

# ハイパーパラメータ
iters_num = 10000
train_size = x_train.shape[0]  ## 消さない
batch_size = 100
learning_rate = 0.1

# 1 エポックあたりの繰り返し数
iter_per_epoch = max(train_size / batch_size, 1)  ## train_size が定義された後に記述

...
```

### 5 章

common/functions.py の softmax 関数は以下のものに書き換えないと gradient_check.py で思うような値が出ない。

``` python
def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)   # オーバーフロー対策
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)
```
