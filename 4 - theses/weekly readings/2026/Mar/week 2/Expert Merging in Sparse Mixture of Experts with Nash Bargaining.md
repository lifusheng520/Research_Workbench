---
title: Expert Merging in Sparse Mixture of Experts with Nash Bargaining

---

# Expert Merging in Sparse Mixture of Experts with Nash Bargaining
Dung V. Nguyen,  Anh T. Nguyen,   Minh H. Nguyen,  Luc Q. Nguyen,  Shiqi Jiang

源代码：https://github.com/anh147/NAMEx

> **文章解决的问题**
> 利用nash博弈来平衡多个专家之间的合并问题，怎么样给权重更公平、更稳定，并通过解nash方程得到的合并权重 $\alpha$ 将多专家参数合成一个 base expert


> **key terms**
1. expert

2. mixture of expert (MoE)

3. sparse MoE：通过为每个输入仅激活一小部分专家来增强可扩展性

4. dense MoE：每个 expert 都对所有 token 计算（全量激活）、 

5. dynamic gating mechanism

6. expert merging

7. utility function：衡量博弈参与者从特定结果中获得的收益

8. 前向（forward pass）： 指的是这一层的 expert（ merged expert $\hat E_m^{(l)}$）要 **向前去处理** 从上一层传过来的输入/hidden state



> **figure 1**
> 对比不同模型，不同专家输出的余弦相似度

> **expert merging**
> 融合旨在将所有专家参数合并到一个统一的模型中，无论是在训练阶段还是推理阶段

> **公式 1**
> 普通 Sparse MoE 的输出怎么来
Sparse Mixture of Experts 的数学描述：
$$ s_i(X; \theta_g) \ge 0, \sum_{i=1}^{N} s_i(X; \theta_g)=1, F(X)= \sum_{i=1}^{N} s_i(X; \theta_g) f_i(X) $$

* $X$：$d$ 维输入向量
* $\theta_g$：决定如何挑选专家的 路由器 (router) 的参数
    例如：一个线性层，一个 MLP，或者带 softmax 的打分模块
    它决定了输入 x 与哪些专家最“匹配”，从而计算出每个专家的激活系数 $s_i$
* $N$：专家的数量
* $s_i()$：叫做gating function，代表路由器给第 i 个 expert 的 权重 / 概率
    根据函数值，判断激活哪些experts（也就是参加计算）
    它决定了第 i 个专家的输出 $f_i(x)$ 在最终结果中占多大比重，即：$s_i(X; \theta_g) f_i(X)$
* $f_i()$：代表第i个专家
* $f_i(X)$：第i个专家的输出
* $F(X)$：对所有输入 X，根据router参数 $\theta_g$ 计算其权重 $s_i$ 与 第i专家输出 $f_i(X)$的乘积（一个加权求和函数，权来源于s(x)函数）


> **公式 2**
>  CAMEx（Curvature-Aware Merging of Experts，中文叫曲率感知专家融合）的“自然梯度式”参数合并公式：
$$\hat{E}_m^{(l)} = E_m^{(l)} + \eta \sum_{i=1}^N M_i^{(l)} \cdot (s_i^{(l)} * \tau_i^{(l)}), \quad\text{where }\ \tau_i^{(l)} = E_i^{(l)} - E_m^{(l)}$$
> 只在当前层做合并

* $E_m^{(l)}$：一个张量，表示 第 $l$ 层 的 基准（base）专家 参数，这是一个共享的expert，即：它每次都会激活并参与计算
 
    一个层内共享的基准/锚点 expert，用于：作为参考点定义 $\tau_i = E_i - E_m$，和跨层传播更新
    
* $η$：步长（step size），一个类似于学习率的系数，用来控制从基准专家迈向合并结果的更新幅度
* $M_i^{(l)}$：第 $l$ 层、第 $i$ 个专家对应的 曲率矩阵（curvature matrix）
* $s_i^{(l)}$：第 $l$ 层 路由器（gating） 为专家 $i$ 产生的 权重/概率（可来自 softmax top‑k 的稀疏权重或期望权重）。
    在 CAMEx 的合并中，它让合并方向与路由策略 对齐 实际使用频率 (作为 缩放系数 调制 $\tau_i^{(l)}$ 的贡献大小)
* $E_i^{(l)}$：第 $l$ 层 第 $i$ 个专家 的参数张量
* $τ_i^{(l)} = E_i^{(l)} − E_m^{(l)}$：第 $l$ 层 第 $i$ 个专家的领域向量（domain‑vector）：它度量专家 $E_i^{(l)}$ 相对基准专家 $E_m^{(l)}$ 的 参数偏移
    
    在数学里，“向量”这个词不仅仅指 $\mathbb{R}^d$ 中的 列向量；它指某个向量空间的元素。而“参数空间”本身就是个线性空间（向量空间）
    
    因此，无论参数是矩阵还是高阶张量，只要你把它们看成“位于同一个参数空间的元素”，它们的差 $E_i^{(l)}-E_m^{(l)}$ 就是这个向量空间里的一个“向量”。这就是论文称其为 domain‑vector（领域向量） 的原因

* $τ_i^{(l)}$：指向“把基准专家 $E_m^{(l)}$ 改成专家 $E_i^{(l)}$所需的参数增量方向

    如果 $\tau_i^{(l)}=0$，说明该专家和基准专家一致，对合并没有推动
    $∥τi(l)∥$ 越大，说明该专家与基准差异越大，潜在贡献/冲突也越大

* $\hat{E}_m^{(l)}$：一个张量，表示第 $l$ 层 合并后的专家权重, 是将基准专家 $E_m^{(l)}$ 按照各专家的“领域向量”与曲率修正合成后的新参数（真正用于前向的 merged expert）

    它是把各专家偏移按路由权重 $s_i$（以及曲率 $M_i$）融合后，真正用于当前层前向处理输入的专家

> **公式 3**
> Expert‑Propagation CAMEx (EP-CAMEx):
> 把第 $l$ 层的 base expert $E_m^{(l)}$ “传播/推进”到下一层得到 $E_m^{(l+1)}$
> $$\begin{cases}
E_{m}^{(l+1)} = E_{m}^{(l)} + \dfrac{\gamma}{N}\displaystyle\sum_{i=1}^{N} M_{i}^{(l)} \cdot \tau_{i}^{(l)},\\[8pt]
\hat{E}_{m}^{(l+1)} = E_{m}^{(l+1)} + \eta \displaystyle\sum_{i=1}^{N} M_{i}^{(l+1)} \cdot \Big(s_{i}^{(l+1)} \ast \tau_{i}^{(l+1)}\Big),
\end{cases}
\qquad
\tau_{i}^{(l)} = E_{i}^{(l)} - E_{m}^{(l)}.$$
> 第一行是传播
> 第二行是在新层（$l + 1$）再结合路由做具体的合并

* $l$：层索引（第 $l$ 个 MoE layer）
* $i$：专家索引（第 $i$ 个 expert）
* $N$：该层专家数量
* $E_i^{(l)}$：第 $l$ 层第 $i$ 个专家的参数（权重），一个张量
* $E_m^{(l)}$：第 $l$ 层的 base expert 参数，一个张量tensor
* $\hat E_m^{(l)}$：第 $l$ 层“合并后用于处理输入”的 merged expert 参数（即最终用于前向计算的那份权重）
* $τ_i^{(l)}=E_i^{(l)}−E_m^{(l)}$：一个向量，表示第 $l$ 层第 i 个专家相对 base expert 的参数偏移（从 base 指向该 expert 的“方向/位移”）
    表示其与基准专家的偏差，并捕捉其对合并过程的独特贡献
* $M_i^{(l)}$：第 $l$ 层第 i 个专家对应的曲率矩阵（curvature matrix）
* $s_i^{(l)}$：第 $l$ 层路由器给专家 i 的权重/概率（反映该专家对当前输入/token 的重要性）
* $γ$：传播（propagation）步长
* $η$：合并（merging）步长

> **公式 4**
> 把专家合并重新解释成一个Nash Bargaining问题
> $$u^* = \arg\max_{u \in S} \prod_{i=1}^{N} (u_i - d_i).$$
> 在所有可行结果 S 中，选择一个结果 $u^*$，使得所有参与者相对于“谈崩时收益” $d_i$ 的收益提升乘积最大

* $N$：参与谈判的“玩家/参与者”数量（这里对应专家数量）
* $S$：可达成协议的效用集合（agreement set / feasible utility set）
    允许选择的“合并更新动作”的范围
* $d$：分歧点/不合作点（disagreement point）。若谈判失败（不达成协议），每个玩家至少得到的基线效用（fallback outcome）
    表示不更新 base expert 的方案
    一个 N 维实向量
* $u$：某个候选协议结果对应的“效用向量”
* $u_i$: 效用向量 u 的第 i 个分量，即玩家 i 在该候选协议下得到的效用(utility)
* $u^*$：**纳什博弈解（Nash Bargaining Solution）** 选出的“最优协议效用向量”

> **公式 5**
> 把多任务学习（MTL）表述成 Nash 博弈中的“效用函数”
> $$u_i(\Delta\theta) = \tau_i^{\top}\Delta\theta $$

* $u_i()$: utility function，表示第 i 个专家的效用：衡量该专家从特定的合并/更新结果中获得的“收益”
    
    通过计算两个向量的 内积（dot product），来度量 $\Delta\theta$ 与 $\tau_i$ 的对齐程度（alignment）
    
    内积大 ⇒ 任务 i 会认为这个更新方向“对任务i很有利”
    内积=0，说明两个向量线性无关，对任务基本没什么用
    内积<0 ⇒ 对任务 i 有害（“对抗/冲突”）

    输入：共享模型参数的 更新方向（update direction）。在 MTL 设定里，大家要“协商”一个共同的更新方向 $\Delta\theta$。
    
* $\Delta\theta$：共享模型参数的 更新方向（update direction）。在 MTL 设定里，大家要“协商”一个共同的更新方向$\Delta\theta$， 使得所有的玩家都受益
    一个vector
* $\tau_i^{\top}$: 任务 i 的损失对模型参数的梯度（task-specific loss gradient with respect to the model parameters）
    表示其与基准专家的偏差，并捕捉其对合并过程的独特贡献


> **公式 6**
> Nash博弈中，共享更新方向的结构：
> $$\Delta \theta = \sum_{i=1}^N \alpha_i \tau_i, \quad \text{where} \quad G^\top G \alpha = 1/\alpha$$
> 把每个任务的梯度方向 $\tau_i$ 按权重 $\alpha_i$ 混合成一个 共享更新方向
> 当任务梯度冲突时，这个线性组合提供一个折中方向，使多个任务都能获得正效用或尽量公平

* $N$：任务数量
* $τ_i$：第 i 个任务的 梯度向量（任务损失对参数的梯度）
* $α$：所有权重组成的向量
* $α_i$：第 i 个任务在最终更新方向中的 权重
* $G=[τ_1,…,τ_N]$：包含了每个任务梯度 $\tau$ 的梯度矩阵
* $G^⊤G$: 生成一个 $N\times N$ 的 Gram 矩阵，其元素是各任务梯度之间的内积$\tau_i^\top\tau_j$，它刻画“任务之间的相似/冲突程度”


> **公式 7**
> 把“跨层传播 base expert 的更新”抽象成一个待求的更新方向 $\Delta E^{(l)}$（第l行），从而可以用 Nash Bargaining 来求出这个方向；而第二行继续做 CAMEx 式的当层合并，得到本层用于前向的 merged expert $\hat E_m^{(l+1)}$
> $$\begin{cases} E_m^{(l+1)} = E_m^{(l)} + \gamma \Delta 𝓔^{(l)} \\
\hat{E}_m^{(l+1)} = E_m^{(l+1)} + \eta \sum_{i=1}^N M_i \cdot (s_i^{(l+1)} * \tau_i^{(l+1)}) \end{cases}$$

* $\Delta 𝓔^{(l)} = ΔE^{(l)}$:
    $𝓔^{(l)}$ 指的是 第 l 层 base expert 的参数张量（或参数集合）（$E_m$） 
    $\Delta 𝓔^{(l)}$ 是对这个 base expert 参数的更新方向，但它的计算会利用所有专家的偏移 $\tau_i^{(l)}$
    
    通过找 $\alpha^{(l)}=[\alpha_1^{(l)},\dots,\alpha_N^{(l)}]$ 来把各个专家的 domain-vector $\tau_i^{(l)}$ 聚合成 $\Delta E^{(l)}$，即$\Delta 𝓔^{(l)}$
    

> **公式 8**
> 怎么样选出最优的更新方向
> $$\arg \max_{\Delta 𝓔 \in B_\epsilon} \sum_{i=1}^N \log(\Delta 𝓔^\top \tau_i)$$
> 
> 把 Nash bargaining 正式改写成专家合并问题之后的目标函数
> 
> 用 **Nash Bargaining** 的思想，为“专家合并更新方向 ($\Delta 𝓔$)”提供一个**原则性优化目标**：选择一个 ($\Delta 𝓔$) 使得所有专家的效用 $(u_i(\Delta 𝓔)=\tau_i^\top \Delta 𝓔)$ 都尽量大，并通过“对数和”形式实现“乘积最大化”的公平性，同时用球约束限制更新幅度
> 
> 数学上等价：$⁡\log\prod = \sum\log$
> 
> Nash 原始形式是最大化乘积（公式 (4)）：$(argmax \prod (u_i-d_i))$。论文在 BEM 中设置分歧点 $(d=0)$，并用线性效用 $(u_i(\Delta E)=\tau_i^\top \Delta E)$，于是目标就是最大化 $(\prod (\Delta E^\top\tau_i))$。对乘积取对数就变成：$\max \sum \log(\Delta E^\top\tau_i)$

* $ΔE = Δ𝓔$: tensor, 表示要寻找的“合并更新方向/更新向量”（用于更新 base expert）
* $B_ϵ$: 允许选择的更新集合（可行域、agreement set），是以 0 为中心、半径为 $\epsilon$ 的球
    这个集合中（set），其中的元素是向量 $\Delta 𝓔$
* $ϵ$: 球约束的半径，控制 $\Delta E$ 最大能有多大（防止更新过猛）
* $τ_i$: 第 i 个专家的 domain-vector（领域向量），表示专家相对 base expert 的偏移方向（在本文中 $\tau_i=E_i-E_m$）
* $ΔE^⊤τ_i$: 这个内积越大，表示更新方向（$Δ𝓔$）让专家的效益越大


> **公式 9**
> 本文的 纳什专家融合（Nash Merging of Experts）
>
> $$\Delta 𝓔^{(l)} = \sum_{i=1}^{N} \alpha_i^{(l)} \tau_i^{(l)}$$
> 
> (9)将纳什博弈解（Nash Bargaining Solution）正式引入专家合并的更新过程中
> NAMEx每一层有两步更新
> $$\begin{cases} E_m^{(l+1)} = E_m^{(l)} + \gamma \sum_{i=1}^N \alpha_i^{(l)} \tau_i^{(l)} \\ \hat{E}_m^{(l+1)} = E_m^{(l+1)} + \eta \sum_{i=1}^N M_i \cdot (s_i^{(l+1)} * \tau_i^{(l+1)}) \end{cases}$$

* 第 1 行（$E_m^{(l+1)}$）：用 Nash 权重 α\alphaα 把专家偏移 τ\tauτ 合成一个“兼顾所有专家”的传播方向，更新 base expert
* 第 2 行（$\hat E_m^{(l+1)}$）：在新层里再用路由权重 sss + 曲率 MMM 做 CAMEx 合并，得到本层可用的 merged expert


> **公式 10**
> $$\alpha_j\|\tau_j\|^2+ \sum_{i\neq j} \alpha_i\tau^\top_i \tau_j= \frac{1}{\alpha_j}$$
> (10)展示了第 j 个专家的纳什合并权重 $α_j$ 是如何由两部分决定的：
> 1. 专家j的自身强度（范数$∥τ_j∥2$）
> 2. 专家 j 与其它专家的“交互/相似/冲突” （$\sum_{i\neq j}\alpha_i \tau_i^\top\tau_j$）
    > 如果向量 $\tau_i, \tau_j$ 正交，说明这两个向量线性无关（即：专家 i 和专家 j 在“更新方向/偏移方向”上几乎不相互影响，既不合作也不冲突）
    > 同理：内积>0，专家之间相互合作 (其他领域向量会帮助第 j 位专家) 
    > 内积<0，冲突，即专家之间存在对抗行为(其他专家会阻碍第 j 位专家)

> 如果所有的 $\tau_j$ 正交，即: $\tau_i^\top \tau_j = 0 \quad (i\neq j)$ 
> 于是, $G^\top G$（Gram 矩阵）变成一个对角矩阵：
> $$G^\top G =
\begin{bmatrix}
\|\tau_1\|^2 & 0 & \cdots & 0 \\
0 & \|\tau_2\|^2 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & \|\tau_N\|^2
\end{bmatrix}$$
> 把它代回 Nash 方程：$(G^⊤Gα)_j = ∥τ_j∥^2α_j = \frac{1}{α_j}$
> 因此得到：$α_j^2 = \frac{1}{∥τ_j∥^2} ⇒ α_j = \frac{1}{∥τ_j∥}, (α∈R_+^N)$


* $\alpha_i, \alpha_j$: Nash 合并中分配给专家 j，i 的权重
* $\tau_i, \tau_j$: 专家的 domain-vector（领域向量），表示该专家参数相对 base expert 的偏移方向，即$\tau_i=E_i-E_m$
* $\|\tau_j\|^2$: 向量的 L2 范数的平方（Square of the L2 norm），就是把对应元素相乘在相加即可
    
    $∥τ_j∥2$ 越大，说明专家 j 的参数整体上离 base 更远（差得更多）


> **公式 11**
> NAMEx-Momentum: Nash Merging with Complex Momentum (NAMEx-动量：纳什合并与复杂动量)
> EP‑CAMEx 的传播受限于层数导致收敛慢，于是引入（Lorraine et al., 2022 的）complex momentum 加速与增强稳定性，数学描述如下：
> $$\begin{cases} 
\mu^{(j+1)} = \beta \mu^{(j)} + \Delta 𝓔^{(j)} \\ 
E_m^{(j+1)} = E_m^{(j)} + \Re(\gamma \mu^{(j+1)}) \\ 
\hat{E}_m^{(l+1)} = E_m^{(l+1)} + \eta \sum_{i=1}^N M_i \cdot (s_i^{(l+1)} * \tau_i^{(l+1)}) 
\end{cases}$$
>
> (11) 把 NAMEx 的更新方向 $\Delta E^{(j)}$ 通过一个 复数动量缓冲 $\mu$ 累积起来，并用 取实部 $\Re(\cdot)$ 的方式更新 base expert $E_m$，以提升在“合作 + 对抗”混合动态下的稳定性与收敛速度；同时保留 CAMEx 式的“路由+曲率合并”来生成本层用于前向的 $\hat E_m$

* $\mu^{(j)}$: 第 j 步的 复数动量缓冲（complex momentum buffer），用于累计历史更新方向
* $β$: 动量系数（momentum coefficient），论文中允许 $\beta \in \mathbb{C}$（复数系数）
* $Δ𝓔^{(j)} = \Delta E^{(j)}$: 一个向量，表示第 j 步 NAMEx 产生的“当前更新方向”（由 Nash 权重 $\alpha$ 聚合 domain-vector 得到）
* $E_m^{(j)}$: 第 j 步（或第 j 层）base expert 的参数（要被传播更新的共享基准专家）
    实参数向量/张量（实现中为参数张量集合，理论上将其向量化为 $\mathbb{R}^d$）
* $γ$: 步长（step size），控制动量更新对 $E_m$ 的影响幅度
* $ℜ(⋅)$: 取实部算子（real-part operator）
* $\hat E_m^{(l+1)}$: 第 l+1 层用于前向计算的 merged expert 参数
* $ℜ(γμ^{(j+1)})$: 首先将动量向量($\mu$)乘以步长 γ，然后提取其实部。
    这是因为神经网络的权重参数通常是实数，必须将复数空间的计算映射回实数空间才能应用更新
* $\eta$: 合并步长（merge step size）
* $M_i$: 第 i 个专家的曲率矩阵/预条件器（curvature matrix）
* $s_i^{(l+1)}$: 路由权重（router weight），表示第 l+1 层把多少权重分给专家 i
* $\tau_i^{(l+1)}$: domain-vector（领域向量），第 l+1 层专家 i 相对 base 的偏移：$\tau_i^{(l+1)}=E_i^{(l+1)}-E_m^{(l+1)}$

第 1 行公式：将当前由纳什博弈计算出的更新方向 $Δ𝓔$ 与带有权重的历史动量信息（$βμ$）相加。由于 β 是复数，这种更新可以利用复数空间的旋转特性来更稳定地处理专家间的竞争动态

第 2 行公式：用“动量累积后的方向”（取实部）更新 base expert，达到加速/稳定传播的目的

第 3 行公式：在当前层把 base expert 变成可用于前向计算的 merged expert， 这一步——它沿用 CAMEx/EP‑CAMEx 的 路由权重 + 曲率矩阵合并项


> **算法 1 Expert Merging via Nash Bargaining**
* 时间复杂度：$O(LN)$
    
    其中：L为SMoE的层数，N为当前层的专家数量
    
* router logits: 第 t 层的路由器（router）对每个 token 应该偏向哪个 expert打出来的一组原始分数
* $H^{(t)} ∈ R^{B×S×N}$:
    B: batch size
    S: sequence length（每个样本里的token数量）
    N: 专家数量
    $H(t)[b,s,i]$： 第 t 层中，第 b 个样本、第 s 个 token，对第 i 个 expert 的 router 原始打分



> **算法 2 NAMEx-Momentum**
* 时间复杂度：$O(LN)$
* $T^{(t)}∈R^{B×S×D}$: 第 t 层的 token 表示序列（token sequence / hidden states），也就是这一层收到的输入特征张量
    B: batch size
    S: sequence length(每个样本中的 token 数)
    D (hidden dimension / embedding dimension): 每个 token 对应的特征维度，也就是每个 token 被表示成多长的向量



> **模型结构**
> 在现有 SMoE / CAMEx / EP‑CAMEx 的基础上引入一个跨层传播的 shared base expert，并用 Nash bargaining 求得每层专家偏移的合并权重 $\alpha$，形成 NAMEx / NAMEx‑Momentum
> 让专家融合的权重更公平、更稳定
> 并 采用NAMEx‑Momentum（动量） 让shared base expert $E_m$ 在层与层之间的更新/收敛过程更快、更稳定


> **实验设置**
> 使用 5 个随机种子
> 
1. language modeling
    * 数据集：WikiText-103
    * 中（216M）小（70M）规模的预训练
    * PPL 越小：模型越不困惑，说明它越会预测文本，语言建模能力越强
    PPL 越大：模型越困惑，说明它对下一个词的预测更不确定
    

2. text classification
    * 数据集：GLUE
    
3. image classification
    * 数据集：ImageNet-1K
    * 并在以下损坏的数据集进一步评估robustness：ImageNet-A, ImageNet-O, and ImageNet-R

4. DeepSeek-MoE (16B) 和 Qwen1.5-MoE (14B) 的零样本和微调
    * DeepSeek-MoE（160 亿参数）
    1 shared expert，63个路由专家
    
    * Qwen1.5-MoE (140 亿参数）
    
