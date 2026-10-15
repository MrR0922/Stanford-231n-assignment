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

