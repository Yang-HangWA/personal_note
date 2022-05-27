# Summary of rs(recommended system)

[papers list from wzhe06](https://github.com/wzhe06/Reco-papers)


#### **done list**
- 1. [[Pinterest] Graph Convolutional Neural Networks for Web-Scale Recommender Systems (Pinterest 2018)](https://github.com/wzhe06/Reco-papers/blob/master/Industry%20Recommender%20System/[Pinterest]%20Graph%20Convolutional%20Neural%20Networks%20for%20Web-Scale%20Recommender%20Systems%20(Pinterest%202018).pdf)
    - https://zhuanlan.zhihu.com/p/60804239

- 2. [[Latent Cross] Latent Cross- Making Use of Context in Recurrent Recommender Systems (Google 2018)](https://github.com/wzhe06/Reco-papers/blob/master/Deep%20Learning%20Recommender%20System/[Latent%20Cross]%20Latent%20Cross-%20Making%20Use%20of%20Context%20in%20Recurrent%20Recommender%20Systems%20(Google%202018).pdf)
   - 其核心思想是希望通过深度模型来模拟并实现在推荐系统中广泛使用的“交叉特征”（Cross Feature）的效果。
论文的主要贡献我们首先来看这篇文章的主要贡献，梳理文章主要解决了一个什么场景下的问题。推荐系统经常需要对当下的场景进行建模，有时候，这些场景被称作“上下文”（Context）。在过去比较传统的方法中，已经有不少方法是探讨如何利用上下文信息进行推荐的，比如使用“张量”（Tensor）的形式进行建模；还有一些方法是利用对时间特性的把握，从而对上下文信息进行处理。近些年，随着深度学习的发展，越来越多的深度学习模型被应用到推荐系统领域中，但还没有直接探究如何在深度学习模型中使用上下文。这篇文章就想在这一方面做一个尝试。这里面有一个比较棘手的问题。过去，这样的上下文常常使用“交叉特性”，也就是两个特征的乘积成为一个新的特征。这样的方法在矩阵分解或者张量分解的模型中得到了非常广泛的使用。然而在深度学习中，过去的经验是不直接使用这样的特性。但是，在上下文非常重要的推荐系统中，不使用交叉特性的的结果，往往就是效果不尽如人意。这篇文章提出了一个叫“隐含交叉”（Latent Cross）的概念，直接作用在嵌入（Embedding）这一层，从而能够在深度模型的架构上模拟出“交叉特性”的效果。论文的核心方法作者们首先探讨了推荐系统中一个常见的特性，那就是利用交叉特性来达到一个“低维”（Low-Rank）的表达方式，这是矩阵分解的一个基本假设。比如每一个评分（Rating）都可以表达成一个用户向量和物品向量的点积。那么，作者们就提出了这样一个问题：作为深度学习的基石，前馈神经网络（Feedforward Neural Network）是否能够很好地模拟这个结构呢？通过模拟和小规模实验，作者们从经验上验证了深度学习的模型其实并不能很好地抓住这样的交叉特性所带来的“低维”表达。实际上，深度学习模型必须依赖更多的层数和更宽的层数，才能得到相同的交叉特性所达到的效果。对于这一点我们或多或少会感到一些意外。同时，作者们在传统的 RNN 上也作了相应的比较，这里就不复述了。得到了这样的结果之后，作者们提出了一个叫作“隐含交叉”的功能。这个功能其实非常直观。传统的深度学习建模，是把多种不同的信息输入直接拼接在一起。“隐含交叉”是让当前的普通输入特性和上下文信息进行乘积，从而直接对“交叉特性”进行建模。这样做的好处是不言而喻的。

- 3. [[xDeepFM] xDeepFM - Combining Explicit and Implicit Feature Interactions for Recommender Systems (USTC 2018)](https://github.com/wzhe06/Reco-papers/blob/master/Deep%20Learning%20Recommender%20System/[xDeepFM]%20xDeepFM%20-%20Combining%20Explicit%20and%20Implicit%20Feature%20Interactions%20for%20Recommender%20Systems%20(USTC%202018).pdf)
    - CIN  https://zhuanlan.zhihu.com/p/57162373
    - CIN 保持了DCN的优点：有限高阶、自动叉乘、参数共享。
    - https://zhuanlan.zhihu.com/p/57162373  相关  DCN：揭秘 Deep & Cross : 如何自动构造高阶交叉特征

- 4. [详解 Wide & Deep 结构背后的动机]( https://zhuanlan.zhihu.com/p/53361519)
    - 显式交叉网络三巨头：DCN、xDeepFM、AutoInt
    - [CTR预估模型：DeepFM/Deep&Cross/xDeepFM/AutoInt代码实战与讲解](https://zhuanlan.zhihu.com/p/109933924)
    - [xDeepFM代码实现](https://github.com/NELSONZHAO/zhihu/blob/master/ctr_models/xDeepFM.ipynb)

    
#### **Embedding**

- [无中生有：论推荐算法中的Embedding思想](https://zhuanlan.zhihu.com/p/320196402)

- [Embedding从入门到专家必读的十篇论文](https://zhuanlan.zhihu.com/p/58805184)

- [详解KDD'2018 best paper—Embedding在Airbnb房源排序中的...](http://wulc.me/2020/06/20/%E3%80%8AReal-time%20Personalization%20using%20Embeddings%20for%20Search%20Ranking%20at%20Airbnb%E3%80%8B%20%E9%98%85%E8%AF%BB%E7%AC%94%E8%AE%B0/)

- [res embedding](https://zhuanlan.zhihu.com/p/141517705)
   - Item Embedding=Central Embedding + Residual embedding

#### 双塔模型
- 百度的双塔模型分别用复杂网络对“用户特征”和“广告特征”进行了embedding化，在最后的交叉层之前，用户特征和广告特征之间没有任何交互，这就形成了两个独立的“塔”，因此称为双塔模型。

![twin tower](https://github.com/Yang-HangWA/DailyNote/blob/master/pic/twin_tower.jpg)


#### 从阿里的User Interest Center看模型线上实时serving方法 

https://zhuanlan.zhihu.com/p/111929212

- 文章总结：这篇文章介绍了阿里妈妈的线上Model Serving的方法User Interest Center，它把模型拆解为线上部分和线下部分，模型复杂的序列结构在线下运行，利用UIC生成和更新Embedding，结果存储在“用户兴趣表达模块”；线上实现模型较为轻量级的MLP部分，使模型能够利用更多的特征进行实时预估。
可以说这是一次机器学习理论和机器学习工程系统完美结合的方案，推荐受困于model serving效率和实时性的团队尝试。