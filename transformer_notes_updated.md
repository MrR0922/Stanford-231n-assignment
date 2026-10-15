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

### 8.1 结构图：从下往上阅读

![Transformer 原论文结构图：右侧为 Decoder](attachments/transformer-architecture-original.png)

图片来源：Vaswani 等，《Attention Is All You Need》，Figure 1。[论文与图片出处](https://arxiv.org/html/1706.03762v7#S3)。图片保留原作者标注，用于个人学习。

右半部分是 Decoder：最下面是右移后的目标序列 embedding 和位置编码；灰色框内从下到上依次为带掩码自注意力、交叉注意力和前馈网络。各模块后面都有 Add & Norm，旁路箭头表示残差连接。图中的 N× 表示重复堆叠的层数，不是本文张量形状中的 batch size N。

图顶部的 Linear 和 Softmax 属于最终词预测，不属于单个 TransformerDecoderLayer。代码训练时通常输出 logits，由损失函数处理 softmax。

### 8.2 输入与维度：tgt 和 memory

本节与 Decoder 代码一致，使用 T 表示目标长度、S 表示上下文长度，D 表示完整 embedding 维度。注意：这与第 3–6 节中 S/T 的角色以及 D 表示每头维度的约定不同；这里每头维度为 D/H。

| 输入 | 形状 | 作用 |
|---|---|---|
| tgt | (N,T,D) | 要被更新的目标文字序列表示 |
| memory | (N,S,D) | 外部上下文，例如投影后的图像特征 |
| tgt_mask | (T,T) | 控制目标位置之间的可见性 |
| 返回值 | (N,T,D) | 融合文字上下文与外部信息后的目标表示 |

`tgt` 是 target 的缩写，这里已是向量表示，不是整数 token 编号。`memory` 不要求一定来自文字编码器；在图像描述中，它可以来自图像特征投影。

### 8.3 第一子层：带掩码的自注意力

Q、K、V 都来自当前 tgt，目标序列内部交换信息：

```python
shortcut = tgt
attn_out = self.self_attn(
    query=tgt, key=tgt, value=tgt, attn_mask=tgt_mask
)
tgt = self.norm_self(shortcut + self.dropout_self(attn_out))
```

注意力分数为 `(N,H,T,T)`。因果 mask 下三角允许关注，上三角屏蔽未来位置；通常允许关注自己，因为输入序列已相对预测目标右移一位。

例如：

```text
输入：<START> 一只 猫
目标：一只    猫   <END>
```

预测“猫”的位置可以看到“<START> 一只”，不能提前看到作为答案的“猫”。

### 8.4 第二子层：交叉注意力

Q 来自上一子层更新后的 tgt；K、V 来自 memory。它回答的是：当前每个文字位置，需要从外部上下文取出什么信息？

```python
shortcut = tgt
attn_out = self.cross_attn(query=tgt, key=memory, value=memory)
tgt = self.norm_cross(shortcut + self.dropout_cross(attn_out))
```

```text
Q：(N,H,T,D/H)
K：(N,H,S,D/H)
V：(N,H,S,D/H)
权重：(N,H,T,S)
合并 head 后的输出：(N,T,D)
```

输出长度跟 query 一致，因此更新的是 tgt。memory 在当前 forward 中不被这段代码覆盖，但梯度仍可通过交叉注意力传回产生 memory 的网络。

`tgt_mask` 的形状是 `(T,T)`，用于自注意力，不能直接当作交叉注意力的 `(T,S)` mask。图像上下文一般允许全部关注；若 memory 含 padding，则需要另外设计相应的屏蔽规则。

### 8.5 第三子层：逐位置前馈网络

每个位置独立经过同一套网络，不在这一步混合不同 token：

$$
\operatorname{FFN}(x)=\operatorname{Linear}_2\left(\operatorname{Dropout}\left(\operatorname{ReLU}(\operatorname{Linear}_1(x))\right)\right)
$$

本作业的内部维度变化为：

```text
(N,T,D) → (N,T,dim_feedforward) → (N,T,D)
```

子层外还有残差和 Dropout：

```python
shortcut = tgt
ffn_out = self.ffn(tgt)
tgt = self.norm_ffn(shortcut + self.dropout_ffn(ffn_out))
```

注意力负责不同位置之间的信息交换，FFN 负责每个位置内部的非线性特征变换。FFN 内部的 Dropout 与残差分支前的 dropout_ffn 是两个不同位置的操作。

### 8.6 Add & Norm、残差与 Post-LN

本作业每个子层都遵循：

$$
x_{next}=\operatorname{LayerNorm}\left(x+\operatorname{Dropout}(\operatorname{Sublayer}(x))\right)
$$

- Add：将子层输出与原输入相加，保留原有表示，并为梯度提供直接路径。
- Norm：nn.LayerNorm(D) 对每个样本、每个 token 的最后 D 维分别归一化，不使用 batch 统计量。
- Dropout：训练时随机丢弃分支中的部分激活，起正则化作用；eval 模式关闭。

归一化位于残差相加之后，所以这是 Post-LN。其他 Transformer 可能采用 Pre-LN，不应混用实现顺序。三个 norm 模块各自具有独立参数。

各子层都输出 `(N,T,D)`，才能与 shortcut 做逐元素相加。代码注释若写返回 `(N,T,W)`，应按实际实现理解为 W=D。

### 8.7 单层完整流程

```python
def forward(self, tgt, memory, tgt_mask=None):
    shortcut = tgt
    tgt = self.self_attn(query=tgt, key=tgt, value=tgt,
                         attn_mask=tgt_mask)
    tgt = self.norm_self(shortcut + self.dropout_self(tgt))

    shortcut = tgt
    tgt = self.cross_attn(query=tgt, key=memory, value=memory)
    tgt = self.norm_cross(shortcut + self.dropout_cross(tgt))

    shortcut = tgt
    tgt = self.ffn(tgt)
    tgt = self.norm_ffn(shortcut + self.dropout_ffn(tgt))
    return tgt
```

整体数据流：

```text
文字表示 tgt
  ↓ Masked Self-Attention：Q/K/V 均来自 tgt
  ↓ Dropout → 残差相加 → LayerNorm
  ↓ Cross-Attention：Q 来自 tgt，K/V 来自 memory ← 图像上下文
  ↓ Dropout → 残差相加 → LayerNorm
  ↓ FFN
  ↓ Dropout → 残差相加 → LayerNorm
更新后的文字表示，形状仍为 (N,T,D)
```

多个 Decoder 层依次更新 tgt，通常查询同一份 memory。self_attn 和 cross_attn 虽然属于同一种 MultiHeadAttention 类，但分别实例化，拥有独立权重。

### 8.8 PyTorch 对象调用与 forward

```python
self.self_attn = MultiHeadAttention(input_dim, num_heads, dropout)
```

这是创建层对象并注册为子模块，不是把类本身赋给变量。随后写 `self.self_attn(...)`，Python 触发继承自 nn.Module 的调用机制，执行该层的 forward，同时处理已注册的 hooks。hooks 是在特定执行时刻自动调用的额外函数，例如打印中间输出形状。

通常使用 `layer(x)`，而不是直接 `layer.forward(x)`，以保留框架的调用机制。

### 8.9 图像描述的训练和生成

```text
图像特征 → 投影 → memory ──────────────────┐
文字前缀 → embedding + 位置编码 → Decoder 堆叠 → 词表 logits
                                 ↑交叉注意力
```

训练时使用右移的真实描述作为输入，预测下一词，并忽略 padding 位置的损失。因果 mask 防止未来答案泄漏，因此可以并行计算所有训练位置。

测试时从起始 token 开始，基于已生成的前缀逐步预测下一词，直到结束标记或最大长度。训练并行不代表自回归生成也能一次生成全部文字。

小数据过拟合实验验证模型能否学会训练样本，不代表泛化良好；Notebook 的该实验给出的训练损失目标为小于 0.05。


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
