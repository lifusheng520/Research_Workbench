---
title: 1. 矩阵和向量的区别，联系

---

# 1. 矩阵和向量的区别，联系
# 2. 矩阵的加减，乘法，除法
# 3. 训练集，测试集，验证集
# 4. batch size, epoch, learning rate

> batch
*  单次传给模型的训练数据单元（训练集）
*  内部样本具有随机性

> batch size
*  划分出来的单次喂给模型的训练单元的大小（样本数量）

> epoch

*  **definition**： 模型已经完成了一次对训练集中所有的样本的训练

*  设置epoch的原因：由于梯度下降过程中，每次的步长（ α ）很小，是缓慢向最小值处靠拢的；所以一个epoch并不能保证损失函数已经彻底/尽可能收敛了。

    *  epoch过大：模型过拟合，泛化能力下降
    *  epoch不足：模型欠拟合，导致模型表现很差（训练集、测试集、推理数据）



# 5. 什么是mlp

> MLP：一种神经网络结构

> MLP结构

* input layer
接收特征值

* hidden layer
    * 一个或多个 fully connected layer
    * 每个隐藏层的每个神经元都和前一层的所有神经元相连
    * 层内神经元会对输入做加权求和，并通过非线性激活函数（如 ReLU、Sigmoid、Tanh）输出。
    * 当前层的输出向量作为下一层的输入，并通过反向传播更新  𝑊  和  𝑏
  
$$ 𝑣_1=𝑓(𝑊𝑥 + 𝑏) $$

$$ 𝑣_2=𝑓(𝑊_2 𝑣_1 + 𝑏_2) $$

$$ 𝑣_𝑖=𝑓(𝑊_𝑖 𝑣_(𝑖−1) + 𝑏_𝑖) $$

𝑓(𝑥) ：激活函数
$𝑊_𝑖$ ：上一层计算完后，更新后的权重矩阵
$b_i$：上一层计算完后，更新后的偏置
$v_i$：输出向量

*  output layer

    输出预测结果（概率 or 回归值）



# 6. 用LLM如何做生成
> 大语言模型做inference过程如下：

$$P(x_1, x_2, \dots, x_T)
= \prod_{t=1}^{T} P(x_t \mid x_1, \dots, x_{t-1})$$

> 本质上：输入一些 token → 模型给出下一个 token 的概率分布 → 选一个 token → 继续预测

## transformer 如何做生成
1. Data Preparation

    * 数据收集 和 预处理
    * 数据集拆分
    
2. tokenization

    *  将输入的字符串（Prompt）切分成若干 Tokens。

      例如：“深度学习很神奇”切成：[“深度学习”, “很”, “神奇”]
      *  然后，查词表，将切分的词变成数字 $[id_1, id_2, ..., id_k]$。

3. embedding
数据转换：
      * 词向量: 根据词的 id 从 Embedding Matrix 中取出 第 id 行。从而得到一个长向量

        矩阵行数为词表的行数，列数取决于设计的模型宽度
    
      * positional encoding：由于 Transformer 的 Attention 机制是并行运算的（diff RNN），无法识别顺序。所以，必须在 embedding时，需要标注vector中每个 id 原本的位置信息，让模型知道“AB”和“BA”的区别。
$$ E=Embedding(x_t)+PositionalEncoding(t)$$

4.   Transformer 处理

>  self-attention

*  数据经过几十层 结构相同、参数不同 的 Transformer Block

      每一层都有自己的一套参数矩阵：$W_Q^{(i)}, W_K^{(i)}, W_V^{(i)}$

* 对所有 历史位置的向量（已有且可见的向量） 做相似度计算，得到一个score向量

* 通过Mask（掩码）保证模型不能看到未来的 token

* 多头注意力（multi-head attention）并行捕捉不同语义，得到新的上下文表示（向量）。

* 经过 softmax 后：得到一个score的概率分布向量/注意力权重向量 （和为1）

* 输出新的上下文感知向量，作为下一层连接层的输入

> 前馈网络（Feed-Forward Neural Network）

每一层的 attention 输出会再经过一个两层 MLP，例如：
$$FFN(x)=W_2  FUN(W_1 x + b_1) + b_2 $$


> 重复几十层

> 输出层

* 产生 Logits（原始分数），通过 softMax(x) 得到下一个 token 的概率分布（向量）, 表示词表中每个 token 的概率。

> 模型根据概率选一个词（贪心、beam全局最优 或 top-k随机）

5. 自回归循环

*  把刚选的新词加入到模型输入向量中
*  将新的向量喂给模型
*  重复第4步，直到选出的新词为结束符 EOS（End of Sentence）

# 7. 用BERT如何做情感分类

BETRT (Bidirectional Encoder Representations from Transformer)

*  **bidirectional：** 基于自编码机制
*  **encoder：** transformer中的encoder部分
*  **representation：** 用于提取高维度的语义特征

1. 数据预处理
    *  tokenization
    *  添加 CLS 和 SEP 标记（token ID）
    *  查词表，将token转为整数（ID）
    *  生成三个向量（Token Embedding、Segment Embedding 和 Position Embedding），将三个向量相加得到 模型的输入向量
$$Input = token + segment + position$$

    * segment embedding：这是因为预训练阶段，BERT是以两个句子作为输入的，segment embedding用来标记这两个句子

        例如：$SE = [0, 0, 0, 0, 1, 1]^T$，0部分为第一个句子，1部分为第二个句子
        
2. Transformer Encoder 进行编码处理
    *  双向自注意力：
      
        因为BERT训练的目的不是为了预测下一个词，而是完成MLM（Masked Language Model）
      
        所以，每个 token 都可以利用完整上下文（token左右两边的其他token）来更新自己的向量表示

        原先的CLS向量 H[CLS] 变成整个句子的压缩语义表示，H[CLS]聚合了整个句子的语义。

        所以情感分类任务只需要拿这个向量即可

    * FFN转换：将每个位置的向量经过前馈神经网络进行非线性变换，使向量成为更高级的语义表示
    
3. 取出 向量$H[CLS]$

4. Classification Head
    *  通过 线性变换 得到 logits
$$logits = W \cdot H_{[CLS]} + b$$ logits中的每个维度代表一种情感类别的评分

    *  再利用 softmax函数 计算可能性，得到一个三维的概率分布向量，向量的每一个维度代表一种情感分类的 概率
    
5. fine-tuning

    *  为了调整 W 和 b，必须计算 损失
$$L  = softmax(logits) - y_i $$

    *  计算 logits 的梯度

    *  得到 logits 的梯度后，利用梯度下降算法 更新 $W$、 $b$ 和 BERT 参数，直到 Loss收敛
    
6. 预测
    *  加载刚刚训练好的模型
    *  将输入的样本进行：分词，加 CLS 和 SEP，然后生成 token id向量
    *  将token id向量输入给BERT，得到一个包含这个样本语义的CLS向量
    *  通过分类头这一步，可以得到这个句子的情感分类概率分布，从而预测这个句子最大概率是怎么样的




# 1. 矩阵和向量的区别，联系
# 2. 矩阵的加减，乘法，除法
# 3. 训练集，测试集，验证集
# 4. batch size, epoch, learning rate

> batch
*  单次传给模型的训练数据单元（训练集）
*  内部样本具有随机性

> batch size
*  划分出来的单次喂给模型的训练单元的大小（样本数量）

> epoch

*  **definition**： 模型已经完成了一次对训练集中所有的样本的训练

*  设置epoch的原因：由于梯度下降过程中，每次的步长（ α ）很小，是缓慢向最小值处靠拢的；所以一个epoch并不能保证损失函数已经彻底/尽可能收敛了。

    *  epoch过大：模型过拟合，泛化能力下降
    *  epoch不足：模型欠拟合，导致模型表现很差（训练集、测试集、推理数据）



# 5. 什么是mlp

> MLP：一种神经网络结构

> MLP结构

* input layer
接收特征值

* hidden layer
    * 一个或多个 fully connected layer
    * 每个隐藏层的每个神经元都和前一层的所有神经元相连
    * 层内神经元会对输入做加权求和，并通过非线性激活函数（如 ReLU、Sigmoid、Tanh）输出。
    * 当前层的输出向量作为下一层的输入，并通过反向传播更新  𝑊  和  𝑏
$$ 𝑣_1=𝑓(𝑊𝑥 + 𝑏) $$
$$ 𝑣_2=𝑓(𝑊_2 𝑣_1 + 𝑏_2) $$
$$ 𝑣_𝑖=𝑓(𝑊_𝑖 𝑣_(𝑖−1) + 𝑏_𝑖) $$
𝑓(𝑥) ：激活函数
$𝑊_𝑖$ ：上一层计算完后，更新后的权重矩阵
$b_i$：上一层计算完后，更新后的偏置
$v_i$：输出向量

*  output layer

    输出预测结果（概率 or 回归值）



# 6. 用LLM如何做生成
> 大语言模型做inference过程如下：

$$P(x_1, x_2, \dots, x_T)
= \prod_{t=1}^{T} P(x_t \mid x_1, \dots, x_{t-1})$$

> 本质上：输入一些 token → 模型给出下一个 token 的概率分布 → 选一个 token → 继续预测

## transformer 如何做生成
1. Data Preparation

    * 数据收集 和 预处理
    * 数据集拆分
    
2. tokenization

    *  将输入的字符串（Prompt）切分成若干 Tokens。

      例如：“深度学习很神奇”切成：[“深度学习”, “很”, “神奇”]
      *  然后，查词表，将切分的词变成数字 $[id_1, id_2, ..., id_k]$。

3. embedding
数据转换：
      * 词向量: 根据词的 id 从 Embedding Matrix 中取出 第 id 行。从而得到一个长向量

        矩阵行数为词表的行数，列数取决于设计的模型宽度
    
      * positional encoding：由于 Transformer 的 Attention 机制是并行运算的（diff RNN），无法识别顺序。所以，必须在 embedding时，需要标注vector中每个 id 原本的位置信息，让模型知道“AB”和“BA”的区别。
$$ E=Embedding(x_t)+PositionalEncoding(t)$$

4.   Transformer 处理

>  self-attention

*  数据经过几十层 结构相同、参数不同 的 Transformer Block

      每一层都有自己的一套参数矩阵：$W_Q^{(i)}, W_K^{(i)}, W_V^{(i)}$

* 对所有 历史位置的向量（已有且可见的向量） 做相似度计算，得到一个score向量

* 通过Mask（掩码）保证模型不能看到未来的 token

* 多头注意力（multi-head attention）并行捕捉不同语义，得到新的上下文表示（向量）。

* 经过 softmax 后：得到一个score的概率分布向量/注意力权重向量 （和为1）

* 输出新的上下文感知向量，作为下一层连接层的输入

> 前馈网络（Feed-Forward Neural Network）

每一层的 attention 输出会再经过一个两层 MLP，例如：
$$FFN(x)=W_2  FUN(W_1 x + b_1) + b_2 $$


> 重复几十层

> 输出层

* 产生 Logits（原始分数），通过 softMax(x) 得到下一个 token 的概率分布（向量）, 表示词表中每个 token 的概率。

> 模型根据概率选一个词（贪心、beam全局最优 或 top-k随机）

5. 自回归循环

*  把刚选的新词加入到模型输入向量中
*  将新的向量喂给模型
*  重复第4步，直到选出的新词为结束符 EOS（End of Sentence）

# 7. 用BERT如何做情感分类

BETRT (Bidirectional Encoder Representations from Transformer)

*  **bidirectional：** 基于自编码机制
*  **encoder：** transformer中的encoder部分
*  **representation：** 用于提取高维度的语义特征

1. 数据预处理
    *  tokenization
    *  添加 CLS 和 SEP 标记（token ID）
    *  查词表，将token转为整数（ID）
    *  生成三个向量（Token Embedding、Segment Embedding 和 Position Embedding），将三个向量相加得到 模型的输入向量
$$Input = token + segment + position$$

    * segment embedding：这是因为预训练阶段，BERT是以两个句子作为输入的，segment embedding用来标记这两个句子

        例如：$SE = [0, 0, 0, 0, 1, 1]^T$，0部分为第一个句子，1部分为第二个句子
        
2. Transformer Encoder 进行编码处理
    *  双向自注意力：
      
        因为BERT训练的目的不是为了预测下一个词，而是完成MLM（Masked Language Model）
      
        所以，每个 token 都可以利用完整上下文（token左右两边的其他token）来更新自己的向量表示

        原先的CLS向量 H[CLS] 变成整个句子的压缩语义表示，H[CLS]聚合了整个句子的语义。

        所以情感分类任务只需要拿这个向量即可

    * FFN转换：将每个位置的向量经过前馈神经网络进行非线性变换，使向量成为更高级的语义表示
    
3. 取出 向量$H[CLS]$

4. Classification Head
    *  通过 线性变换 得到 logits
$$logits = W \cdot H_{[CLS]} + b$$ logits中的每个维度代表一种情感类别的评分

    *  再利用 softmax函数 计算可能性，得到一个三维的概率分布向量，向量的每一个维度代表一种情感分类的 概率
    
5. fine-tuning

    *  为了调整 W 和 b，必须计算 损失
$$L  = softmax(logits) - y_i $$

    *  计算 logits 的梯度

    *  得到 logits 的梯度后，利用梯度下降算法 更新 $W$、 $b$ 和 BERT 参数，直到 Loss收敛
    
6. 预测
    *  加载刚刚训练好的模型
    *  将输入的样本进行：分词，加 CLS 和 SEP，然后生成 token id向量
    *  将token id向量输入给BERT，得到一个包含这个样本语义的CLS向量
    *  通过分类头这一步，可以得到这个句子的情感分类概率分布，从而预测这个句子最大概率是怎么样的




