# Adam 优化器

Adam（Adaptive Moment Estimation）可以理解为结合了两种优化思想：

- **Momentum**：记录梯度的移动平均，提供更稳定的更新方向。
- **RMSProp**：记录梯度平方的移动平均，为每个参数自适应调整步长。

## 更新公式

### 一阶矩

$$
m_t = \beta_1m_{t-1} + (1-\beta_1)dw_t
$$

### 二阶矩

$$
v_t = \beta_2v_{t-1} + (1-\beta_2)(dw_t)^2
$$

### 偏差修正

$$
\hat m_t = \frac{m_t}{1-\beta_1^t}
$$

$$
\hat v_t = \frac{v_t}{1-\beta_2^t}
$$

### 更新权重

$$
w_{t+1} = w_t - \eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
$$

## 参数含义

- $w_t$：当前权重。
- $dw_t$：当前权重的梯度，即 $\frac{\partial L}{\partial w_t}$。
- $m_t$：梯度的一阶矩，即梯度的指数移动平均。
- $v_t$：梯度的二阶矩，即梯度平方的指数移动平均。
- $\beta_1$：一阶矩衰减率，通常取 `0.9`。
- $\beta_2$：二阶矩衰减率，通常取 `0.999`。
- $\eta$：学习率 `learning_rate`，Adam 通常使用 `1e-3`。
- $\epsilon$：防止除零的小常数，通常取 `1e-8`。
- $t$：参数更新次数，从 `1` 开始参与偏差修正。

## 第一步：更新迭代次数

```python
t = config["t"] + 1
```

必须先将 `t` 加一，因为偏差修正需要计算：

```python
1 - beta1**t
1 - beta2**t
```

如果 $t=0$，两个表达式的结果都是零，会造成除零。

## 第二步：更新一阶矩 $m$

```python
m = beta1 * m + (1 - beta1) * dw
```

对应公式：

$$
m_t = \beta_1m_{t-1} + (1-\beta_1)dw_t
$$

$m$ 类似 Momentum 中的速度，保存近期梯度的平均方向。若 $\beta_1=0.9$：

```text
新 m = 0.9 × 旧 m + 0.1 × 当前梯度
```

这样可以减少单个 mini-batch 带来的梯度噪声。

## 第三步：更新二阶矩 $v$

```python
v = beta2 * v + (1 - beta2) * dw**2
```

对应公式：

$$
v_t = \beta_2v_{t-1} + (1-\beta_2)(dw_t)^2
$$

$v$ 类似 RMSProp 中的 `cache`，记录近期梯度的大小。若 $\beta_2=0.999$：

```text
新 v = 0.999 × 旧 v + 0.001 × 当前梯度平方
```

## 第四步：进行偏差修正

```python
m_hat = m / (1 - beta1**t)
v_hat = v / (1 - beta2**t)
```

$m$ 和 $v$ 的初始值通常都是零：

```python
m = np.zeros_like(w)
v = np.zeros_like(w)
```

因此，训练初期的移动平均会偏向零。以第一次更新为例：

$$
m_1=(1-\beta_1)dw_1
$$

当 $\beta_1=0.9$ 时：

$$
m_1=0.1dw_1
$$

经过偏差修正：

$$
\hat m_1=\frac{0.1dw_1}{1-0.9}=dw_1
$$

这样就消除了初始值为零带来的偏差。

## 第五步：更新权重

```python
next_w = (
    w
    - learning_rate
    * m_hat
    / (np.sqrt(v_hat) + epsilon)
)
```

其中：

- `m_hat` 决定经过平滑后的更新方向。
- `sqrt(v_hat)` 根据近期梯度大小调整每个参数的步长。
- `epsilon` 防止分母为零。

## CS231n 完整实现

```python
def adam(w, dw, config=None):
    if config is None:
        config = {}

    config.setdefault("learning_rate", 1e-3)
    config.setdefault("beta1", 0.9)
    config.setdefault("beta2", 0.999)
    config.setdefault("epsilon", 1e-8)
    config.setdefault("m", np.zeros_like(w))
    config.setdefault("v", np.zeros_like(w))
    config.setdefault("t", 0)

    beta1 = config["beta1"]
    beta2 = config["beta2"]
    learning_rate = config["learning_rate"]
    epsilon = config["epsilon"]

    m = config["m"]
    v = config["v"]
    t = config["t"] + 1

    # 更新一阶矩和二阶矩
    m = beta1 * m + (1 - beta1) * dw
    v = beta2 * v + (1 - beta2) * dw**2

    # 偏差修正
    m_hat = m / (1 - beta1**t)
    v_hat = v / (1 - beta2**t)

    # 更新权重
    next_w = (
        w
        - learning_rate
        * m_hat
        / (np.sqrt(v_hat) + epsilon)
    )

    # 保存未经偏差修正的状态
    config["m"] = m
    config["v"] = v
    config["t"] = t

    return next_w, config
```

## 执行顺序

```text
t 加 1
→ 更新 m
→ 更新 v
→ 对 m 和 v 进行偏差修正
→ 更新 w
→ 保存未经修正的 m、v 和新的 t
```

> [!warning] 注意
> 不要把经过偏差修正的 $\hat m_t$ 和 $\hat v_t$ 保存到 `config`。`config` 应保存未经修正的 `m` 和 `v`，以便下一轮继续计算指数移动平均。

> [!summary] 核心思想
> Adam 使用一阶矩稳定更新方向，并使用二阶矩自适应调整每个参数的步长；偏差修正则消除训练初期由零初始化造成的估计偏差。
