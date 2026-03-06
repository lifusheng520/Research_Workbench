# MLP如何更新每层的权重矩阵 $W$ ?

> 背景：
> 一个单层的神经网络本质上就是一个线性模型 $y = Wx + b$
> 那么：直接采用梯度下降公式更新它的权重 W 即可
> 通过不断迭代更新 W，最终可达到让损失函数最小化，让模型收敛
> 其中(根据矩阵乘法的规则)：
>> $W$ 的 **行数** 取决于 输出 的个数；即，有多少个神经元（输出点））
>> 例如：

$$
\begin{bmatrix}
任务 & 输出个数 & W行数 \\
回归 & 1 & 1行 \\
二分类 (Sigmoid) & 1 & 1 \\
二分类 (Softmax) & 2 & 2 \\
数字识别 & 10 & 10 \\
LLM预测下一个词 & 30,000+ (词表大小) & 30,000+ 行
\end{bmatrix}
$$

>> $W$ 的 **列数** 取决于 输入 的个数（即特征的个数）
>
> 在处理多样本时:
> 主流深度学习框架（如 PyTorch, TensorFlow）和论文通常采用 $Y = XW^T + b$ 
> 或者直接定义 $W$ 为 (输入 $\times$ 输出) 的形式
> 总而言之，就是为了让矩阵乘法成立
> 最后，通过通过一个非线性激活函数 $\sigma$（如 ReLU 或 Sigmoid）就得到了下一层神经网络的输出
> 如果神经网络只有单层（输入层 $\rightarrow$ 输出层），且单输出。没有经过非线性激活函数处理，那么这种结构通常被称为：
> 线性层 (Linear Layer)
> 全连接层 (Fully Connected Layer)
> 稠密层 (Dense Layer)


**那么多层神经网络如何最小化损失函数，让模型收敛？**

答案：
Rumelhart D E, Hinton G E, Williams R J. Learning representations by back-propagating errors[J]. nature, 1986, 323(6088): 533-536.

> 这篇文章解决思路：
> 利用复合导数求导法，从损失函数出发，逐层计算参数（W 和 b）的变化率（梯度）
> 然后根据变化率更新 W 和 b

**算法：**
1. 第一层（hidden layer）
$$z^1 = W^1 x + b^1 \quad
v^1 = \sigma(z^1)$$

第 $i$ 层 hidden layer：
$$z^{(i)} = W^{(i)} v^{(i - 1)} + b^{(i)} \quad
v^{(i)} = \sigma(z^{(i)})
$$

输出层 output layer：
$$z^{(n)} = W^{(n)} v^{(n - 1)} + b^{(n)} $$
$$ \hat y = \sigma(z^{(i)}) $$

$\sigma$ ：激活函数 
$W^{(i)}$：上一层计算完后，更新后的权重矩阵 
$b^{(i)}$：上一层计算完后，更新后的偏置 
$v^{(i)}$：当前 hidden layer 的输出（向量）
$\hat y$：最终的预测输出（向量）

2. 损失函数：假设用MSE 
$$ L = \frac{1}{2} (\hat y - y)^2 $$

3.函数关系：
$$L \rightarrow v^{(i)} \rightarrow z^{(i)} 
\rightarrow \cdots
\rightarrow v^{(1)} \rightarrow z^{(1)} \rightarrow W^{(1)}$$

4. 计算 W 的变化率
从损失函数可得：
$$L = \frac{1}{2}(\sigma(z^{(n)}) - y)^2$$
因此：$z^{(n)}$ 是 $W^{(n)}$ 的一个函数

$$\frac{\partial L}{\partial W^{(n)}} = \frac{\partial L}{\partial z^{(n)}} \cdot \frac{\partial z^{(n)}}{\partial W^{(n)}} = \delta^{(n)} (v^{(n-1)})^T$$

其中：
$$ \delta^{(n)} = \frac{\partial (\frac{1}{2}(\sigma(z^{(n)}) - y)^2)}{\partial z^{(n)}} = (\sigma(z^{(n)}) - y) \cdot \frac{\partial \sigma(z^{(n)})}{\partial z^{(n)}}$$

理解：
**权重 $W$ 的梯度 = 函数 $z$ 的变化率 × 上一层的输入**







