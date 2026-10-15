# Transformer 学习笔记

> 来源：CS231n assignment3 / Transformer_Captioning.ipynb。
> 重点：注意力、多头计算、位置编码、图像描述与 Vision Transformer。

## 1. 为什么使用 Transformer

RNN 按序列顺序计算，训练难以充分并行，长距离信息也需要经过多个时间步传递。Transformer 使用注意力，让不同位置直接交换信息，并在训练时并行处理各位置。自回归生成时仍需要逐步生成 token。

## 2. 点积注意力

给定一个查询向量 $q$、键向量 $k_j$ 和值向量 $v_j$：

$$
\alpha_j = \frac{\exp(k_j^\top q)}{\sum_{r=1}^{L}\exp(k_r^\top q)},\qquad
c = \sum_{j=1}^{L}\alpha_jv_j
$$

- Query：当前需要寻找什么信息。
- Key：用来与 query 匹配的表示。
- Value：匹配后实际汇总的信息。
- $\alpha_j$：注意力权重，非负且总和为 1。
- $c$：所有 value 的加权汇总，维度与单个 value 相同。

加权求和可以同时利用多个位置的信息，而不必只选一个位置。权重接近 1 时，也近似于选择对应的 value。

## 3. Q、K、V 从哪里来

采用 token 向量按行排列的约定：

$$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V
$$

$W_Q,W_K,W_V$ 是训练得到的权重矩阵，$Q,K,V$ 是由输入计算出的张量。若把单个向量按列表示，也可以写作 $q_i=W_Qx_i$；这时权重矩阵的形状和转置约定相应改变。

PyTorch 中：

```python
self.query = nn.Linear(E, E)  # 创建带可学习权重和偏置的层
q = self.query(query)        # 调用该层：q = query @ weight.T + bias
```

每个 token 使用相同的线性层。输入输出均为 `(N, S, E)`，形状不变，但数值和表示方式改变。

### 自注意力与交叉注意力

自注意力：query、key、value 来自同一序列，因此序列长度相同；经过不同投影后，三者通常不相同。

交叉注意力：query 来自当前序列，key、value 来自上下文，序列长度可以不同。例如文字查询图像特征。

本文统一记号：

| 符号 | 含义 |
|---|---|
| $N$ | batch size |
| $S$ | query 序列长度 |
| $T$ | key/value 序列长度 |
| $E$ | embedding 维度 |
| $H$ | head 数量 |
| $D=E/H$ | 每个 head 的维度 |

不要仅凭 source/target 名称判断长度，应以函数参数的实际形状为准。

## 4. 多头缩放点积注意力

对于自注意力，第 $i$ 个 head 为：

$$
Y_i=\operatorname{softmax}\left(
\frac{(XW_{Q,i})(XW_{K,i})^\top}{\sqrt{D}}
\right)(XW_{V,i})
$$

每个投影矩阵形状为 $E\times D$；每个 head 可以从完整输入中提取不同的信息。不同 head 的参数通常独立，但并不保证学到的内容完全不同。

### 为什么除以平方根

如果 query 和 key 的各分量具有近似单位方差，$D$ 个乘积之和的方差会随 $D$ 增大。除以 $\sqrt D$ 控制分数尺度，避免 softmax 过于集中、梯度变小。

### 为什么 embedding 可以拆分

实现通常先用一次线性层同时计算所有 head，再拆分结果：

```python
q = self.query(query)  # (N, S, E)
q = q.reshape(N, S, H, D).permute(0, 2, 1, 3)
# (N, H, S, D)
```

例如长度 8 的向量拆成 2 个 head，每个 4 维：

```text
[a,b,c,d,e,f,g,h] → [[a,b,c,d], [e,f,g,h]]
```

reshape 不产生新数值，只改变分组方式，要求 $E=HD$。因为拆分发生在线性投影之后，每个 head 的表示都可以依赖原 embedding 的全部维度，并非只能看到原始输入的一小段。

### reshape、permute 和矩阵乘法

- `reshape`：重新组织形状，元素总数不变。
- `permute(0, 2, 1, 3)`：把 `(N, S, H, D)` 调整为 `(N, H, S, D)`。
- `transpose(-2, -1)`：仅交换最后两个维度。
- `torch.matmul` / `@`：最后两维做矩阵乘法，前面的维度按广播规则分组处理。

```text
Q                  (N,H,S,D)
K.transpose(-2,-1)  (N,H,D,T)
匹配分数             (N,H,S,T)
```

注意：NumPy 三维数组的 `.T` 会反转全部轴，不能直接当作这里的最后两维转置。

### 为什么权重与 value 做矩阵乘法

设权重为 $P$，则：

$$
Y_{s,d}=\sum_{t=1}^{T}P_{s,t}V_{t,d}
$$

这恰好是对每个 query 的 value 加权求和：

```text
权重 (N,H,S,T) @ value (N,H,T,D) → 输出 (N,H,S,D)
```

`F.softmax(scores, dim=-1)` 沿 key 的维度计算权重，使每个 query 对所有 key 的权重总和为 1。

## 5. 因果掩码与 Dropout

### 因果掩码

生成文本时，当前位置不能看到未来 token。先单独创建下三角 mask：

```python
mask = torch.tril(torch.ones(S, S, device=query.device))
```

```text
1 0 0
1 1 0
1 1 1
```

行表示 query，列表示 key/value。1 允许关注，0 屏蔽。

```python
if attn_mask is not None:
    scores = scores.masked_fill(attn_mask == 0, float('-inf'))
```

不需要把 scores 放进 mask。mask 只是按位置提供规则；`(S,T)` 的 mask 可以广播到 `(N,H,S,T)`。

被屏蔽位置的分数变为负无穷，softmax 后因 $\exp(-\infty)=0$，权重变为 0。直接把分数设为 0 无法实现屏蔽，因为 $\exp(0)=1$。每行至少需要一个有效位置，否则整行负无穷会使 softmax 产生 NaN。

### 注意力 Dropout

Notebook 在 softmax 后、加权求和前应用 Dropout：

$$
Y_i=\operatorname{dropout}\left(\operatorname{softmax}(\text{scores}_i)\right)V_i
$$

训练时随机丢弃部分注意力连接，减少对固定连接的过度依赖。保留的权重按保留概率缩放，使期望保持不变；Dropout 后每行权重之和不保证为 1。测试时关闭 Dropout。

## 6. 拼接各个 head 与输出投影

$$
Y=\operatorname{Concat}(Y_1,\ldots,Y_H)A
$$

这里沿特征维度拼接，不是重复相加。拼接后恢复 $E$ 维，再通过可学习矩阵 $A\in\mathbb R^{E\times E}$ 混合各个 head 的信息。代码的线性层还包含偏置。

```python
output = output.permute(0, 2, 1, 3).reshape(N, S, E)
output = self.proj(output)
```

### 多头注意力计算流程

```python
q = self.query(query).reshape(N, S, H, D).permute(0, 2, 1, 3)
k = self.key(key).reshape(N, T, H, D).permute(0, 2, 1, 3)
v = self.value(value).reshape(N, T, H, D).permute(0, 2, 1, 3)
scores = (q @ k.transpose(-2, -1)) / math.sqrt(D)
if attn_mask is not None:
    scores = scores.masked_fill(attn_mask == 0, float('-inf'))
weights = self.attn_drop(F.softmax(scores, dim=-1))
output = weights @ v
output = self.proj(output.permute(0, 2, 1, 3).reshape(N, S, E))
```

无论有没有 mask，都需要执行 softmax 和输出计算；这些步骤不能放进 mask 的 `if` 分支内。

## 7. 位置编码

没有位置信息时，自注意力本身无法区分输入的先后顺序，因此将位置编码与 token embedding 相加：

$$
Z=X+P
$$

对位置 $p$ 和成对的维度 $2r,2r+1$：

$$
P_{p,2r}=\sin\left(\frac{p}{10000^{2r/E}}\right),\qquad
P_{p,2r+1}=\cos\left(\frac{p}{10000^{2r/E}}\right)
$$

不同维度使用不同频率。$P$ 与 embedding 维度一致，相加后仍为 `(N,S,E)`，同一套位置编码广播到 batch 中的各个样本。

## 8. Transformer Decoder 与图像描述

Notebook 的 decoder block 包含：

1. 带因果 mask 的自注意力：处理当前文字序列。
2. 交叉注意力：文字表示作为 query，图像特征表示作为 key/value。
3. 逐位置前馈网络：对每个位置独立使用相同的网络。

典型前馈网络：

$$
\operatorname{FFN}(x)=W_2\operatorname{ReLU}(W_1x+b_1)+b_2
$$

注意力负责位置之间的信息交换；FFN 负责每个位置内部的特征变换。实际 block 还结合残差连接、Dropout 和 LayerNorm；归一化在子层前还是后取决于实现。

图像描述流程：

```text
图像特征 → 投影为上下文表示 ───────────┐
文字 token → embedding + 位置编码 → decoder → 词表分数
                                     ↑交叉注意力
```

训练时输入描述的前缀，预测右移一位的下一个词，并忽略 padding 位置的损失。因果 mask 允许并行计算各位置的训练输出，又防止看到未来答案。

测试时从起始 token 开始，将已生成的前缀送入模型，逐步预测下一个 token，直到结束标记或最大长度。

小数据上的过拟合实验用于验证模型能够学会训练样本，不代表泛化良好。Notebook 给出的训练损失目标是小于 0.05。

## 9. Vision Transformer（ViT）

### Patch embedding

将图像切成互不重叠的固定大小 patch，每个 patch 展平并线性投影为 $E$ 维向量。

设图像为 $C\times H_{img}\times W_{img}$，patch 边长为 $p$，尺寸可整除时：

$$
L=\frac{H_{img}W_{img}}{p^2},\qquad
\text{每个 patch 的原始维度}=Cp^2
$$

输出形状为 `(N,L,E)`，再加位置编码。

### Encoder 与分类

Encoder block 由自注意力和逐位置 FFN 构成，配合残差和归一化。图像分类不要求因果限制，所有 patch 可以互相关注。

本作业对最终所有 patch 表示做平均池化，再预测类别；这里不使用额外的 CLS token。作业采用一维正弦位置编码，也可以使用其他位置编码方案。

### 小数据集上的局限

CNN 自带局部连接、权重共享等归纳偏置。标准 ViT 缺少同样强的图像局部性约束，需要更多数据学习这些规律，因此在小数据集上可能落后于 CNN。

常见改善方向：大数据预训练后微调、数据增强、适度正则化，以及引入局部结构或卷积的混合架构。

### 注意力计算成本

忽略 QKV 与输出投影，注意力矩阵和加权汇总的主要计算量为：

$$
O(L^2E)
$$

| 单独改变 | 注意力计算量变化 |
|---|---|
| embedding 维度翻倍 | 约 2 倍 |
| 图像高度、宽度均翻倍 | patch 数变为 4 倍，计算约 16 倍 |
| 正方形 patch 的边长翻倍 | patch 数变为 1/4，计算约 1/16 |
| 层数翻倍 | 全部注意力层总计算约 2 倍 |

以上保持其他条件不变，并假设 patch 可以整齐划分图像。每个 head 的注意力矩阵存储量随 $L^2$ 增长。

## 10. 实现检查清单

- embedding 维度必须能被 head 数整除。
- 先做 QKV 投影，再 reshape 分头。
- 匹配分数形状是 `(N,H,S,T)`；只有自注意力时 $S=T$。
- softmax 沿最后一维执行。
- mask 在 softmax 前应用，Dropout 在 softmax 后应用。
- 合并 head 前先把维度调整为 `(N,S,H,D)`。
- 测试时调用 `model.eval()` 关闭 Dropout。
- Obsidian 公式使用 `$...$` 或 `$$...$$`，避免多余的结束符；不依赖 `\label` 引用。
