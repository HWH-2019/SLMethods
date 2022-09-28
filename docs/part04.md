## 朴素贝叶斯法
- 是基于贝叶斯定理与特征条件独立假设的分类方法
- 对于给定训练数据集，首先基于特征条件独立假设学习输入输出的联合概率分布；
- 然后基于此模型，对给定的输入 $x$,利用贝叶斯定理求出后验概率最大的输出 $y$


## 朴素贝叶斯法的学习与分类

### 基本方法
- 朴素贝叶斯法对条件概率分布作了条件独立性假设

$$
\begin{equation*}
	\begin{split}
  P(X=x|Y=c_k)
  & = P(X^{(1)} = x^{(1)},...,X^{(n)} = x^{(n)}|Y=c_k) \\
  & = \prod_{j=1}^{n}P(X^{(j)} = x^{(j)}|Y = c_k) \\
  \end{split}
\end{equation*}
$$

- 朴素贝叶斯法实际上学习到生成数据的机制，属于生成模型
- 条件独立假设等于是说用于分类的特征在类确定的条件下都是条件独立的
- 这一假设使朴素贝叶斯法变得简单，但有时会牺牲一定的分类准确率
- 后验概率计算根据贝叶斯定理进行：

$$
  P(X=x|Y=c_k) = \frac{P(X = x|Y = c_k)P(Y = c_k)}{\sum_kP(X=x|Y=c_k)P(Y = c_k)}
$$

- 两者代入后得朴素贝叶斯法分类的基本公式

$$
 P(X=x|Y=c_k) = \frac{P(Y = c_k)\prod_jP(X^{(j)} = x^{(j)}|Y = c_k)}{\sum_kP(Y = c_k)\prod_jP(X^{(j)} = x^{(j)}|Y = c_k)}
$$

- 贝叶斯分类器可表示为

$$
y = f(x) = \mathop{\mathrm{argmax}}\limits_{c_k}\frac{P(Y = c_k)\prod_jP(X^{(j)} = x^{(j)}|Y = c_k)}{\sum_kP(Y = c_k)\prod_jP(X^{(j)} = x^{(j)}|Y = c_k)}
$$

- 等价于

$$
y = f(x) = \mathop{\mathrm{argmax}}\limits_{c_k}{P(Y = c_k)\prod_jP(X^{(j)} = x^{(j)}|Y = c_k)}
$$

### 后验概率最大化的含义

- 朴素贝叶斯法将实例分到后验概率最大的类中，这等价于期望风险最小化。
- 根据期望风险最小化准则可以得到后验概率最大化准则

$$
f(x) = \mathop{\mathrm{argmax}}\limits_{c_k}P(c_k|X=x)
$$

## 朴素贝叶斯法的参数估计

### 极大似然估计
- 先验概率 $P(Y=c_k)$ 的极大似然估计

$$
P(Y=c_k) = \frac{\sum_{i=1}^{N}I(y_i = c_k)}{N}, k=1,2,...,K
$$

- 条件概率 $P(X^{(j)} = a_{jl}|Y = c_k)$ 的极大似然估计

$$
\begin{equation*}
	\begin{split}
  P(X^{(j)}
  = a_{jl}|Y = c_k) = \frac{\sum_{i=1}^{N}I(x_i^{(j)} = a_{jl},y_i = c_k)}{\sum_{i=1}^{N}I(y_i = c_k)} \\
  j = 1,2,...,n; l = 1,2,...,S_j; k = 1,2,...,K \\
  \end{split}
\end{equation*}
$$

### 学习与分类算法

- 计算先验概率及条件概率
- 对于给定的实例 $x = (x^{(1)},x^{(1)},...,x^{(1)})^T$, 计算

$$
P(Y = c_k)\prod_{j=1}^nP(X^{(j)} = x^{(j)}|Y = c_k), k=1,2,...,K
$$

- 确定实例 $x$ 的类

### 贝叶斯估计
- 用极大似然估计可能会出现所要估计的概率值为 0 的情况，这时会影响到后验概率的计算结果，使分类产生偏差
- 解决上述问题的方法是采用贝叶斯估计
- 条件概率的贝叶斯估计是

$$
P_\lambda(X^{(j)} = a_{jl}|Y = c_k) = \frac{\sum_{i=1}^{N}I(x_i^{(j)} = a_{jl},y_i = c_k)+\lambda}{\sum_{i=1}^{N}I(y_i = c_k) + S_j\lambda}
$$

- 上式中 $\lambda \geq 0$
- 等价于在随机变量各个取值的频数上赋予一个正数 $\lambda > 0$
- 当 $\lambda = 0$ 时就是极大似然估计
- 当 $\lambda = 1$ 时就是拉普拉斯平滑
