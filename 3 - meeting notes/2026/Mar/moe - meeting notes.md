---
title: moe - meeting notes

---

moe
CCFA 先了解MOE
然后 细看
**融合** 多个专家 = 1    顶会找
quit 砍掉

研究方向：**model merging**后看

MoDES: Accelerating Mixture-of-Experts Multimodal Large Language Models via Dynamic Expert Skipping
2026-2 cvpr


3-13
1.有哪些做模型融合的方法？
模型融合时；**是多个人1+1=1**（无base expert），还是1+5=1（有base expert）
2.模型融合的时候，他们的参数会发生什么变化，模型融合之后，新模型的**参数是变大？还是变小？或=？**
3. 当模型融合后，在谁身上做测试？怎么证明融合有效？
4. 新的数据（数学+物理）和旧的数据集（物理，数学），无交集的新的数据集一定会有domain shift
如果有domain shift怎么办？
**查阅文章** 有方法可做


3-20
找问题：how to find a new problem setting? （假设、约束、目标、data）
    输入，输出
motivation？-> method

需要做的：
1. 查一下：有没有人这样做过？（**看截图IB** -〉实际上也是在skipping）
2. 有无人做**OT做merging**？routing已经有人做
3. 看别人的方法？怎么做的？
4. 我的话，要么skip要么merge
