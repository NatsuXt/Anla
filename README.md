# 🛸 Project Anla：全原生复数 AGI 核心引擎技术档案

**版本**: Kernel v1.5 / Transformer Integration Alpha

**日期**: 2026-02-03

**状态**: 核心模块验证完成，Transformer 集成测试中（遭遇能量爆炸问题）

---

## 1. 项目宏观愿景与哲学

Anla 旨在构建一个不依赖传统实数自动微分框架的**全复数神经网络**。它不仅是将数值改为复数，而是试图模拟物理动力学系统。

* **几何优先**：利用复数域特有的相位旋转、干涉和共形变换能力，捕捉序列数据中的拓扑结构和相对位置关系。
* **物理动力学**：摒弃全局优化的 SGD/Adam，采用**类 Hebbian (Hebbian-like)** 的局部更新规则。网络被视为一个能量耗散系统。
* **原生反向传播**：手动推导并实现 Wirtinger 导数（复数梯度），直接传递**复数误差向量**而非标量 Loss 梯度。

---

## 2. 核心架构详述 (Architecture Specification)

### 2.1 引擎内核：手动微分 (The Engine)

* **基类**: `Anla.core.base_layer.ComplexLayer`
* **机制**:

  * **Forward**: 缓存输入 `input_cache`。
  * **Manual Backward**: 接收后层传来的复数误差向量 $\delta_{out} \in \mathbb{C}^{N}$，计算并返回前传误差 $\delta_{in}$，同时原地更新参数。
* **设计约束**: 显式禁止使用 PyTorch 的 `autograd`，以确保梯度流完全受控于复数逻辑。

### 2.2 线性层与更新规则 (Linear Layer)

* **实现**: `Anla.layers.linear.ComplexLinear`
* **前向**: $Y = X W^T + B$ (复数矩阵乘法)
* **反向更新 (Hebbian)**:
  $$ \Delta W = \eta \cdot (\delta_{out}^T \cdot X^*) $$

  * **物理含义**: 权重的更新方向取决于误差向量与输入共轭的积。共轭操作 $X^*$ 用于抵消输入的相位，提取相对相位差。
* **稳定性修正**:

  * **Weight Decay (阻尼)**: 必须引入，否则纯 Hebbian 规则会导致能量无限积聚（共振灾难）。
  * **Gradient Clipping (限幅)**: 限制单次更新的模长，防止相位翻转。
  * **3D 适配**: 已修正 `.t()` 错误，支持 `(Batch, Seq, Dim)` 的自动展平与聚合。

### 2.3 激活函数 (Activation Function)

* **实现**: `Anla.layers.activation.PhaseTwist`
* **当前版本**: **V1 (Linear Twist)**
* **公式**:
  $$ f(z) = \tanh(|z|) \cdot e^{i(\theta + \gamma \cdot |z|)} $$

  * $\gamma$ (Gamma) 是可学习参数。
* **设计演变史**:

  * *V1 (Linear)*: 初始设计。配合 RMSNorm 效果最佳。
  * *V2 (Log-Saturation)*: 曾尝试 $\log(1+|z|)$ 防止爆炸，但导致梯度刚度过大，已废弃。
  * *V3 (Bidirectional)*: 曾尝试让相位影响模长，导致相位拟合崩溃，已废弃。
* **功能**: 引入“强度-相位耦合”。赋予网络非全纯性，使其具备处理异或 (XOR) 等逻辑冲突的能力。

### 2.4 归一化 (Normalization) —— 系统的锚

* **实现**: `Anla.layers.normalization.ComplexRMSNorm`
* **公式**:
  $$ z_{out} = \frac{z}{\sqrt{\text{mean}(|z|^2) + \epsilon}} \cdot S $$

  * $S$ 是可学习实数缩放因子。
* **作用**: **至关重要**。它强制将层间信号的模长约束在单位球附近。没有它，Hebbian 网络会瞬间发散。它解耦了“能量”与“相位信息”。

### 2.5 注意力机制 (Complex Attention)

* **实现**: `Anla.layers.attention.ComplexAttention`
* **核心逻辑**: **埃尔米特内积 (Hermitian Inner Product)**
  $$ \text{Score} = \text{Re}(Q \cdot K^H) $$

  * 取实部意味着：关注相位的一致性 ($|Q||K|\cos(\Delta\theta)$)。
  * **天然位置编码**: 复数乘法天然包含旋转相对性，无需额外的 RoPE。
* **反向传播**: 已完成复杂的 Wirtinger 导数推导，包含 Softmax 穿过实部取操作的梯度流。

---

## 3. 实验验证记录 (Proven Milestones)

我们通过三个关键实验验证了内核的完备性：

1. **$z \to z^2$ (流形逼近)**

   * **结果**: 相位映射图呈现完美的 $2\times$ 频率锯齿波。
   * **结论**: 证明了 PhaseTwist 可以通过叠加逼近乘法频率变换，理解拓扑缠绕。

2. **XOR 象限分类 (逻辑完备)**

   * **结果**: 决策边界呈现清晰的棋盘格 (Checkerboard)。
   * **结论**: 证明了网络能解决线性不可分问题，利用相位扭曲实现了空间的拓扑折叠。

3. **导航与相变 (Dynamics)**

   * **结果**: 发现低学习率 (0.001) 诱导非线性 Gamma 上升，高学习率 (0.005) 导致线性坍缩。
   * **结论**: 揭示了 Anla 的自适应性。为了保留智能（非线性），应倾向于使用较低的学习率。

---

## 4. 当前阻碍与危机 (Critical Issue: Transformer Explosion)

**当前任务**: `Anla/tests/test_transformer.py` (序列预测)
**状态**: **失败/待修复**

### 现象描述

在 Transformer 集成测试中，Loss 在 Epoch 0 极低 (~0.05)，随后迅速飙升至 ~200 并持续震荡。

### 根本原因分析：能量正反馈环 (Positive Feedback Loop)

这是一个复数 Hebbian 网络特有的病态结构问题：

1. **共享源头**: Input 和 Target 都来自同一个 Embedding 矩阵。
2. **Hebbian 特性**: 更新规则 $\Delta W \propto \delta \cdot X^*$ 倾向于拉长权重向量（增加能量）。
3. **循环爆炸**:

   * 梯度更新 $\to$ Embedding 向量变长。
   * 下一轮 Target 变长 $\to$ 距离误差 ($|T-P|^2$) 自然变大。
   * Loss 变大 $\to$ 梯度变大 $\to$ Embedding 更新更猛烈。
   * 系统陷入“为了追逐更远的目标而跑得更快，导致目标跑得更远”的死循环。

---

## 5. 下一步行动路线图 (Action Plan)

请接手方按以下步骤推进：

### 步骤 1: 实施流形约束 (Manifold Constraint)

**这是当务之急。** 必须打破 Embedding 的能量正反馈。

* **操作**: 修改 `test_transformer.py`。
* **逻辑**: 在每次 `manual_backward` 后，强制将 `model.embed.weight` 归一化到单位球。

  ```python
  model.embed.weight.data = complex_normalize(model.embed.weight.data)
  ```

  (或者除以模长: `w / (|w| + eps)`)

### 步骤 2: 验证 Transformer 收敛

* **目标**: 观察 Loss 是否由升转降，或者稳定下降。
* **参数建议**: 学习率设为 `0.001` (低能态以保护非线性)，Weight Decay `1e-4`。

### 步骤 3: 扩展序列任务

如果步骤 2 成功，尝试更有说服力的任务：

* **Copy Task**: 输入 `A B C A B C`，预测下一个。需要 Attention 极其精准地搬运相位。
* **Reverse Task**: 输入 `A B C`，输出 `C B A`。

### 步骤 4: 架构封装 (Refactoring)

目前的训练循环中手动调用各层 `backward` 过于繁琐且易错。

* **建议**: 创建 `AnlaModel` 类，内部维护层列表，提供统一的 `backward(delta)` 接口，自动倒序调用各层的 manual_backward。

---

**最终备注**:
Anla 是一个极其敏感的动力学系统。它像生物神经元一样，不仅传输信息，还传输能量。**控制能量（归一化/阻尼）是控制智能的前提。** 请务必在每一步修改中关注模长（Magnitude）的变化。

---

**测试**:
在Anla文件夹外层以模块形式调用即可:

```python
python -m Anla.main
```

或同理：tests目录下的若干测试代码
