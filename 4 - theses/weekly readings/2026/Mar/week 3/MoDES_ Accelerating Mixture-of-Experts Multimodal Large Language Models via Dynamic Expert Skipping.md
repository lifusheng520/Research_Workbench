---
title: 'MoDES: Accelerating Mixture-of-Experts Multimodal Large Language Models via Dynamic Expert Skipping'

---

# MoDES: Accelerating Mixture-of-Experts Multimodal Large Language Models via Dynamic Expert Skipping

源代码： https://github.com/ModelTC/MoDES

> 这篇文章做的是 expert skipping，在不改动模型参数的情况下，在推理时动态地决定“哪些参数不参与计算”

> 
> 文章方法：动态专家跳过（Dynamic Expert Skipping）
> 采用全局调制局部门控法（Globally-Modulated Local Gating） 来 保留浅层关键层中的专家；而可以更积极地跳过深层、影响力较小的层中的专家。
> 采用文本token的阈值 和 视觉token的阈值，来分别控制每种模态的专家跳过程度。


> key terms
1. MLLM 与 LLM


> **公式 1&2**
> 高级MLLM中所应用的 MOE 层作为FFN的数学描述：
> $$\pi_m^{(l)} = \frac{\exp\!\left(r_m^{(l)}\right)}{\sum_{\hat m = 1}^{M} \exp\!\left(r_{\hat m}^{(l)}\right)}, \quad y^{(l+1)} = \sum_{m \in S^{(l)}} \pi_m^{(l)} \cdot \mathrm{Expert}_m^{(l)}\!\left(x^{(l)}\right)$$
> 解释：
> 在第 $l$ 层，路由器先给每个 expert 一个分数 $r_m^{(l)}$；然后用 softmax 把这些分数变成归一化概率 $\pi_m^{(l)}$；概率越大，说明这个 expert 对当前 token 越重要、越可能被选中
> 
> 在第 $l$ 个 MoE 层中，只对被选中的 top-k experts 进行计算，然后按 routing probabilities $\pi_m^{(l)}$ 对这些 expert 的输出进行加权聚合，得到该层的最终输出 $y^{(l+1)}$
> 
> **这篇文章认为**：这种与层无关的规则忽略了不同层专家对全局贡献（即对最终输出的影响）的不平衡 


*  $\pi_m^{(l)}$: 第 l 层中，第 m 个 expert 对当前 token 的 routing probability, 表示这个 expert 被当前 token 选中/重视的相对程度
*  $r_m^{(l)}$: 第 $l$ 层 router 给第 m 个 expert 输出的 routing logit。定义 $r^{(l)}=\{r_1^{(l)},\dots,r_M^{(l)}\}$ 是一组 routing logits
*  $\exp()$: $\exp(x)=e^x$
*  $Expert_m^{(l)}(x^{(l)})$: 第 $l$ 层的第 m 个专家，对输入第$l$层的输入x的输出。每个 expert 本质上是一个 MLP
*  $S^{(l)}$: 表示路由概率最大的前 k 位专家的索引集合



> **公式 3 & 4**
> 全局调制局部门控（Globally-Modulated Local Gating）
> $$\text{for: } i \in S^{(l)},\quad s_i^{(l)} = \alpha^{(l)} \cdot \pi_i^{(l)}, \quad \alpha^{(l)} = \frac{1}{N} \sum_{j=1}^{N} D_{\mathrm{KL}}\!\left(\mathrm{prob}_j || \mathrm{prob}^{(l)}_j\right)$$
> 
> 先在离线阶段计算每一层的全局重要性 $α^{(l)}$，推理阶段直接使用预先算好的 $\alpha^{(l)}$ 即可。它通过比较“原模型输出分布”和“跳过该层 experts 后的模型输出分布”之间的 KL divergence(KL散度) 来量化这一层的重要程度；差异越大，说明这一层越重要，于是 $\alpha^{(l)}$ 越大
> 
> 然后，在推理阶段，把局部的路由概率 $\pi_i^{(l)}$ 和全局的层重要性 $\alpha^{(l)}$ 结合起来，得到更准确的 expert importance score $s_i^{(l)}$，用于判断当前 token 在当前层中哪些 experts 可以被跳过，如果低于 threshold，会跳过这个 expert 的计算

* $s_i^{(l)}$: 第 $l$ 层、第 $i$ 个 expert 对于当前 token 的 importance score（重要性分数）
* $\alpha^{(l)}$: 第 $l$ 层的 globally-modulated factor（全局调制因子），用来反映 这一层的 experts 对最终预测的整体影响有多大
* $N$: 校准数据集 $C=\{c_1,\dots,c_N\}$ 的样本数
* $C$: 校准数据（calibration data）是从现有数据集里抽样得到的一小部分样本集合，用来做离线校准（offline calibration）
* $prob_j$: 原始模型在第 j 个校准样本上的 输出概率分布 向量
* $prob_j^{(l)}$: 在“跳过第 $l$ 层 experts”的修改模型上，第 j 个校准样本对应的 输出概率分布 向量
* $D_{KL}(\cdot||\cdot)$: Kullback–Leibler divergence（KL 散度），用于衡量两个概率分布之间的差异



> **公式 5、6 & 7**
> 双模态阈值法（Dual-Modality Thresholding）
> 
> $$\left\{ \text{Expert}_i^{(l)} \mid s_i^{(l)} < \tau_{\text{t}} \cdot \mathbb{I}_{\text{t}} + \tau_{\text{v}} \cdot \mathbb{I}_{\text{v}} \right\} \quad (5)$$
> $$\min_{\tau_{\text{t}} \in \mathcal{B},\, \tau_{\text{v}} \in \mathcal{B}} f(\tau_{\text{t}}, \tau_{\text{v}}) \quad \text{s.t.}
\quad g(\tau_{\text{t}}, \tau_{\text{v}}) \ge \rho \quad (6)$$
> $$p_{(q)} = \min \left\{ p \in \{1, \ldots, D\} \mid g(\tau^{(p)}, \tau^{(q)}) \ge \rho \right\} \quad (7)$$
>
>  （7）定义了一个 frontier 索引函数：固定一个 $q$（也就是先固定一个候选阈值 $\tau^{(q)}$ 作为其中一个模态的阈值），然后在另一个模态的候选阈值轴上的 $D$ 个候选里找，找一个最小的索引 $p$，使得这对阈值$(τ^{(p)},τ^{(q)})$ 以及满足跳过比例$g(τ^{(p)},τ^{(q)})≥ρ$。这样就不需要枚举所有 $D^2$ 个阈值组合，而只需要沿着 frontier 搜索，大大降低复杂度。论文说其时间复杂度从朴素的 $O(ND^2)$ 降到 $O(ND)$。
>  
>  选择最优的文本阈值 $\tau_t$ 和视觉阈值 $\tau_v$，最终结果min优化后，输出一对最优阈值标量 $(\tau_t, \tau_v)$
> 
> 公式5 根据 token 的模态（文本 / 视觉），用对应的阈值筛掉“不重要”的专家，输出当前 token 在第 $l$ 层应该被跳过的专家集合

* $Expert_i^{(l)}$: 第 $l$ 个 MoE 层中的第 i 个专家
* $τ_t, τ_v$: 分别是，文本 token 的阈值 和 视觉 token 的阈值
* $\mathbb{I}_t$: 文本 token 指示函数（indicator function），判断当前 token 是否为文本 token
* $\mathcal{B} = \{ \tau^{(1)}, \cdots ,\tau^{(D)} \}$: 候选阈值搜索网格（search grid set），而且 $τ(1)<τ(2)<⋯<τ(D)$
* $D$: 指的是搜索空间 $\mathcal{B}$ 中候选阈值（Candidate Thresholds）的总数量
* $f(τ_t, τ_v)$: 原始模型输出分布与按照公式 (5) 跳过专家后的模型输出分布之间的平均 KL 散度; 衡量“性能损失 / 输出偏移”。值越小，说明跳专家后的模型越接近原模型
* $g(τ_t, τ_v)$: 被跳过专家的比例（fraction of experts that are skipped）；衡量效率提升有多大。值越大，跳过越多。
* $\rho$: 目标跳过比例（target skipping ratio）。这是效率约束，比如希望至少跳过 80% 专家，就可取 $\rho=0.8$
* $s.t.$ : subject to的缩写
* $p^{(q)}$: 在固定 $q$ 时，对应的 frontier 位置索引。表示“满足约束的最小可行 $p$”。
* $q$: 候选阈值索引
* $p$: 候选索引范围内 $\{1, \cdots, D \}$的一个值
* $\tau^{(p)}$: 候选阈值集合 $\mathcal{B} = \{ \tau^{(1)}, \tau^{(2)}, \cdots, \tau^{(D)} \}$ 中第 $p$ 个阈值
* $g(\tau^{(p)}, \tau^{(q)})$: 在这对候选阈值下的专家跳过比例。





> **figure 2**
> 作者认为：浅层专家远比深层专家重要，且层级贡献不平衡
> 当 k 值减小时（向左移动），Layer 1-10 的曲线（浅层）下降最为剧烈。相比之下，Layer 17-26 的曲线（深层）非常平缓，甚至在 k 值较小时仍能保持较高的分数
> 来源解释了这种现象的原因——浅层产生的误差会被后续层不断放大，从而导致最终输出的“误差爆炸”（error explosion）


> **figure 3**
> left: 文本和视觉token 在各层之间存在一致的分布差异
> 
> middle：同一个 token，在进入 FFN/MoE 层之前的表示 和 经过 FFN/MoE 层之后的表示 之间的余弦相似度。纵轴是这一层里该模态 token 的相似度统计值。
> 文本 Token 的相似度较低且波动大，说明专家层对文本信息的更新幅度显著更大
> 
> right： 进一步追问“为什么视觉 token 被 FFN 改得更少”。 表示 tokens 和 FFN 权重之间的角度，视觉 Token 向量与权重矩阵中的这些行向量在几何空间上几乎是垂直的，所以矩阵相乘后的结果（更新量）就会非常小（FFN 对视觉 token 的作用更弱）
> 所以，在做 expert skipping 时，文本 token 通常应该被 更谨慎地（少）跳过，而视觉 token 可以 更激进地（多）跳过。
> 
> 这直接启发了作者设计 DMT（双模态阈值处理）：在推理时，给视觉 Token 设置更激进的跳过阈值，让它们更多地“绕过”专家层，从而在不损害精度的前提下大幅提速


> **实验**
1. 对比试验

对比其它 3 个多模态MOE大模型，证明：
 a. 在尽量高的专家跳过比例（skipping ratio）下，MoDES 能比已有 expert skipping 方法更好地保住多模态模型性能(accuracy)
 b. 同时带来实际推理加速

* 对比的 MoE MLLM 模型：Kimi-VL；Qwen3-VL-MoE；InternVL-3.5。



2. 消融试验

>> MoDES 的第一个关键模块是 GMLG（Globally-Modulated Local Gating）
* 以前的多模态MoE大模型只看当前 token 的 router 分数 $\pi_i^{(l)}$，而不看这一层本身对最终输出的重要性 $\alpha^{(l)}$，通过消融试验来证明：
    * 只用局部路由概率：是“层无关”的；它忽略了不同层的重要性程度
    * 引入 $\alpha^{(l)}$ 后：能让跳过策略对 关键层（浅层）更保守（skipping的专家数量少一些），对非关键层（深层）更激进 （skipping的专家数量多）
* 结果：GMLG 存在时，MoDES 的专家重要性估计更准确，因此在同样跳过率下能保住更多性能。


>> MoDES 的第二个核心模块是 DMT（Dual-Modality Thresholding）。它为 文本 和 视觉 两类 token 分别设置阈值

消融试验要验证的是：
如果把文本 token 和视觉 token 混在一起，用同一个阈值，会不会比“双模态分开阈值”效果差？
实验对比：
* 统一阈值
* 双模态阈值

