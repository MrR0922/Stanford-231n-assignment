# RMSProp

## 更新公式

RMSProp 使用梯度平方的指数移动平均，动态调整每个参数的实际学习率：

$$
\mathrm{cache}_t = \rho\,\mathrm{cache}_{t-1} + (1-\rho)(dw_t)^2
$$

$$
w_{t+1} = w_t - \eta\frac{dw_t}{\sqrt{\mathrm{cache}_t}+\epsilon}
$$

这里的 `cache` 保存梯度平方的指数移动平均：

- 梯度长期较大的位置：分母较大，更新幅度减小。
- 梯度长期较小的位置：分母较小，更新幅度相对增大。

## 参数含义

- $w_t$：当前的模型参数，例如某层的权重 `W1`。
- $dw_t$：损失函数对 $w_t$ 的梯度，决定参数更新的方向。
- $\eta$：学习率 `learning_rate`，控制总体更新步长。
- $\rho$：衰减率 `decay_rate`，控制 `cache` 对历史梯度的记忆程度，通常取 `0.99`。
- $\mathrm{cache}_t$：截至当前时刻的梯度平方指数移动平均。
- $\epsilon$：防止除零的小常数，通常取 `1e-8`。
- $w_{t+1}$：更新后的模型参数。

## 梯度平方的指数移动平均

$$
\mathrm{cache}_t = \rho\,\mathrm{cache}_{t-1} + (1-\rho)(dw_t)^2
$$

其中，$\rho\,\mathrm{cache}_{t-1}$ 保留过去的梯度信息。若：

```python
decay_rate = 0.99
```

就会保留旧 `cache` 的 99%：

```python
0.99 * cache
```

而 $(1-\rho)(dw_t)^2$ 加入当前梯度的信息：

```python
0.01 * dw**2
```

梯度平方有两个主要作用：

- 消除梯度的正负号，只记录梯度大小。
- 让较大的梯度在 `cache` 中产生更明显的影响。

对应代码为：

```python
cache = (
    decay_rate * cache
    + (1 - decay_rate) * dw**2
)
```

## 自适应参数更新

$$
w_{t+1} = w_t - \eta\frac{dw_t}{\sqrt{\mathrm{cache}_t}+\epsilon}
$$

其中：

$$
\frac{dw_t}{\sqrt{\mathrm{cache}_t}+\epsilon}
$$

是经过自适应缩放后的梯度。

当某个参数近期的梯度一直较大时：

```text
cache 较大
→ sqrt(cache) 较大
→ 缩放后的梯度变小
→ 参数更新幅度减小
```

当某个参数近期的梯度一直较小时：

```text
cache 较小
→ sqrt(cache) 较小
→ 缩放后的梯度相对变大
→ 参数更新幅度相对增大
```

因此，RMSProp 可以为不同参数分别调节实际更新步长。

## 为什么需要开平方？

`cache` 保存的是梯度平方 $(dw)^2$，其数量级也是梯度的平方。由于：

$$
\sqrt{(dw)^2}=|dw|
$$

开平方后可以恢复到与梯度相近的数量级。

## 为什么需要加 $\epsilon$？

当梯度一直为零时，`cache` 也可能为零。直接计算：

```python
dw / np.sqrt(cache)
```

可能发生除零。因此在分母中加入一个很小的常数：

```python
epsilon = 1e-8
dw / (np.sqrt(cache) + epsilon)
```

## NumPy 实现

```python
rho = config["decay_rate"]
eta = config["learning_rate"]
epsilon = config["epsilon"]
cache = config["cache"]

cache = rho * cache + (1 - rho) * dw**2
next_w = w - eta * dw / (np.sqrt(cache) + epsilon)

config["cache"] = cache
```

> [!summary] 核心思想
> 普通 SGD 对所有参数使用同一个学习率；RMSProp 根据每个参数近期的梯度大小，分别调整其实际更新步长。
