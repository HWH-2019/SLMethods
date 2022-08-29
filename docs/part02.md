## 感知机
- 感知机是二分类的线性分类模型，其输入为实例的特征向量，输出为实例的类别，取 +1 和 -1 二值
- 感知机对应于输入空间（特征空间）中将实例划分为正负两类的超分离平面，属于判别模型
- 感知机学习旨在求出将训练数据进行线性划分的超分离平面

## 感知机模型
假设输入空间（特征空间）是 $X \subseteq \mathbb{R}^n$  ， 输出空间是 $Y = {+1,-1}$。输入 $x \in X$ 表示实例的特征向量，对应于输入空间（特征空间）的点； 输出 $y \in Y$ 表示实例的类别。 由输入空间到输出空间的如下函数：
$$ f(x) = sign(\omega \cdot x + b)$$
称为感知机。其中 $\omega$ 和 $b$ 为感知机模型参数，$\omega \in \mathbb{R}^n$ 叫做权值 (weight) 或权值向量 (weight vector)，$b \in \mathbb{R}$ 叫做偏置 (bias)，$\omega \cdot x$ 表示 $\omega$ 和 $x$ 的内积。$sign$ 是符号函数，即

$$sign(x) = \begin{cases}
+1, & x \geq 0 \\
-1, & x < 0
\end{cases}$$


## 感知机学习策略
- 假设训练数据集是线性可分的。
- 感知机学习的目标是求得一个能够将训练集正实例点和负实例点完全正确分开的分离超平面。
- 为了找出这样的超平面，即确定感知机模型参数 $\omega$ , $b$，需要确定一个学习策略，即定义（经验）损失函数井将损失函数极小化。

### 损失函数
- 误分类点的总数（不是参数 $\omega$ , $b$ 的连续可导函数，不易优化）
- 误分类点到超平面 $S$ 的总距离（感知机采用，公式表示如下，$\parallel \omega \parallel$ 是 $\omega$ 的 $L_2$ 范数）

$$-\frac{1}{\parallel \omega \parallel}\sum_{x_i \in M}y_i(\omega \cdot x_i + b)$$

- 不考虑 $\frac{1}{\parallel \omega \parallel}$ ，就得到感知机学习的损失函数（公式表示如下,这个损失函数就是感知机学习的经验风险函数）

$$L(\omega,b) = -\sum_{x_i \in M}y_i(\omega \cdot x_i + b)$$

- 感知机学习的策略是在假设空间中选取使损失函数最小的模型参数 $\omega$ , $b$，即感知机模型

## 感知机学习算法
- 感知机学习问题转化为求解损失函数的最优化问题，最优化的方法是随机梯度下降法
- 具体算法包括原始形式和对偶形式

### 感知机学习算法的原始形式
- 假设误分类点集合 $M$ 是固定的，那么损失函数 $L(\omega,b)$ 的梯度公式为

$$
\bigtriangledown_{\omega}L(\omega,b) = -\sum_{x_i \in M}y_ix_i
$$

$$
\bigtriangledown_bL(\omega,b) = -\sum_{x_i \in M}y_i
$$

- 随机选取一个误分类点 $(x_i,y_i)$，对 $\omega$ , $b$ 进行更新

$$
\omega \gets \omega + \eta y_ix_i
$$

$$
b \gets b + \eta y_i
$$

其中 $\eta(0 < \eta \leq 1)$ 是步长，在统计学习 中又称为学习率 (learning rate)。这样，通过迭代可以期待损失函数 $L(\omega,b)$ 不断减小，直到为 $0$。

#### 算法流程
- 选取初值 $\omega_0$ , $b_0$;
- 在训练集中选取数据 $(x_i,y_i)$;
- 如果 $y_i(\omega \cdot x_i + b) \leq 0$,

$$
\omega \gets \omega + \eta y_ix_i
$$

$$
b \gets b + \eta y_i
$$

- 转至第二步，直至训练集中没有误分类点。

#### 直观解释
- 当一个实例点被误分类，即位于分离超平面的错误一侧时，则调整 $\omega$ , $b$ 的值，使分离超平面向该误分类点的一侧移动，以减少该误分类点与超平面间的距离，直至超平面越过该误分类点使其被正确分类。

### 感知学习算法的对偶形式
- 基本想法：将 $\omega$ 和 $b$ 表示为实例 $x_i$ 和标记 $y_i$ 的线性组合的形式，通过求解其系数二求得 $\omega$ 和 $b$
- 不失一般性，在原始形式算法中可假设初始值 $\omega_0$ , $b_0$ 均为0
- 对误分类点 $(x_i,y_i)$ 通过如下公式，逐步修改 $\omega$ , $b$

$$
\omega \gets \omega + \eta y_ix_i
$$

$$
b \gets b + \eta y_i
$$

- 假设修改 $n$ 次，则 $\omega$ , $b$ 关于 $(x_i,y_i)$ 的增量分别是 $\alpha_iy_ix_i$ 和 $\alpha_iy_i$，这里 $\alpha_i = n_i\eta$。
- 这样从学习过程中不难看出，最后学习到的 $\omega$ , $b$ 可以分别表示为

$$
\omega = \sum^{N}_{i=1}\alpha_iy_ix_i
$$

$$
b = \sum^{N}_{i=1}\alpha_iy_i
$$

- 实例点更新次数越多，意味着它距离分离超平面越近，也就越难正确分类（这样的实例对学习结果影响最大）

#### 算法流程
- $\alpha \gets 0, b \gets 0$;
- 在训练集中选取数据 $(x_i,y_i)$;
- 如果 $y_i(\sum^{N}_{i=1}\alpha_iy_ix_i \cdot x_i + b) \leq 0$,

$$
\alpha_i \gets \alpha_i + \eta
$$

$$
b \gets b + \eta + \eta y_i
$$

- 转至第二步，直至训练集中没有误分类点。
