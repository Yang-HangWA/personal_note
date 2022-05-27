# Summary of transformers&bert


#### **for understanding**
- [为什么Transformer 需要进行Multi-head Attention？ - 知乎](https://www.zhihu.com/question/341222779)
    - `multihead`就是类似于`CNN`的多个卷积核，不同的头负责不同的分工，而有的任务可能需要多个`head`协作来完成

- [详解Transformer （Attention Is All You Need） - 知乎](https://zhuanlan.zhihu.com/p/48508221)
    - [外文博客](http://jalammar.github.io/illustrated-transformer/)


- 作者采用`Attention`机制的原因是考虑到`RNN`（或者`LSTM`，`GRU`等）的计算限制为是顺序的，也就是说`RNN`相关算法只能从左向右依次计算或者从右向左依次计算，这种机制带来了两个问题：
	- 1. 时间片<img src="http://latex.codecogs.com/svg.latex?t" title="http://latex.codecogs.com/svg.latex?t" />的计算依赖<img src="http://latex.codecogs.com/svg.latex?t-1" title="http://latex.codecogs.com/svg.latex?t-1" />时刻的计算结果，这样限制了模型的并行能力；
	- 2. 顺序计算的过程中信息会丢失，尽管`LSTM`等门机制的结构一定程度上缓解了长期依赖的问题，但是对于特别长期的依赖现象,`LSTM`依旧无能为力。

- `Transformer`的提出解决了上面两个问题，首先它使用了`Attention`机制，将序列中的任意两个位置之间的距离是缩小为一个常量；其次它不是类似`RNN`的顺序结构，因此具有更好的并行性，符合现有的`GPU`框架。那么`Query，Key，Value`是什么意思呢？它们在Attention的计算中扮演着什么角色呢？我们先看一下`Attention`的计算方法，整个过程可以分成7步：
	- 1. 如上文，将输入单词转化成嵌入向量；
	- 2. 根据嵌入向量得到 <img src="http://latex.codecogs.com/svg.latex?q,k,v" title="http://latex.codecogs.com/svg.latex?q,k,v" /> 三个向量；
	- 3. 为每个向量计算一个score：  ；
	- 4. 为了梯度的稳定，Transformer使用了score归一化，即除以 <img src="http://latex.codecogs.com/svg.latex?\sqrt{q_{k}}" title="http://latex.codecogs.com/svg.latex?\sqrt{q_{k}}" />  ；
	5. 对score施以softmax激活函数；
	6. softmax点乘Value值<img src="http://latex.codecogs.com/svg.latex?v" title="http://latex.codecogs.com/svg.latex?v" />，得到加权的每个输入向量的评分<img src="http://latex.codecogs.com/svg.latex?v" title="http://latex.codecogs.com/svg.latex?v" />；
	7. 相加之后得到最终的输出结果 <img src="http://latex.codecogs.com/svg.latex?z:z=\sum&space;v" title="http://latex.codecogs.com/svg.latex?z:z=\sum v" /> 。


在self-attention需要强调的最后一点是其采用了残差网络 [5]中的short-cut结构，目的当然是解决深度学习中的退化问题
Query，Key，Value的概念取自于信息检索系统，举个简单的搜索的例子来说。当你在某电商平台搜索某件商品（年轻女士冬季穿的红色薄款羽绒服）时，你在搜索引擎上输入的内容便是Query，然后搜索引擎根据Query为你匹配Key（例如商品的种类，颜色，描述等），然后根据Query和Key的相似度得到匹配的内容（Value)。


#### **example**
[Transformer 在美团搜索排序中的实践](https://tech.meituan.com/2020/04/16/transformer-in-meituan.html)


### 解释
- Transformer结构
   - Transformer模型也是使用经典的encoer-decoder架构，encoder，decoder分别有6层Transformer block结构。
   - encoder中的transformer block由两部分组成，第一部分是一个multi-head self-attention mechanism; 第二部分是一个position-wise feed-forward network，是一个全连接层.
   - 两个部分，都有一个 残差连接(residual connection)，然后接着一个Layer Normalization。
   - decoder也有6个，但是每个包含三部分，即第一个部分是multi-head self-attention mechanism，第二部分是multi-head context-attention mechanism，
   第三部分是一个position-wise feed-forward network。这三个部分的每一个，都有一个残差连接，后接一个Layer Normalization。

- atention通俗解释，attention是指，对于某个时刻的输出y，它在输入x上各个部分的注意力。这个注意力实际上可以理解为权重。

- 「self-attention」，也叫 「intra-attention」，是一种通过自身和自身相关联的 attention 机制，从而得到一个更好的 representation 来表达自身，self-attention 可以看成一般 attention 的一种特殊情况。在 self-attention 中， ，序列中的每个单词(token)和该序列中其余单词(token)进行 attention 计算。self-attention 的特点在于「无视词(token)之间的距离直接计算依赖关系，从而能够学习到序列的内部结构」，实现起来也比较简单，值得注意的是，在后续一些论文中，self-attention 可以当成一个层和 RNN，CNN 等配合使用，并且成功应用到其他 NLP 任务。

- Why Multi-head Attention
    - 原论文中说到进行 Multi-head Attention 的原因是将模型分为多个头，形成多个子空间，可以让模型去关注不同方面的信息，最后再将各个方面的信息综合起来。其实直观上也可以想到，如果自己设计这样的一个模型，必然也不会只做一次 attention，多次 attention 综合的结果至少能够起到增强模型的作用，也可以类比 CNN 中同时使用「多个卷积核」的作用，直观上讲，多头的注意力「有助于网络捕捉到更丰富的特征/信息」。


- 点乘attention缩放因子的原因([最佳解释](https://zhuanlan.zhihu.com/p/69697467))： 文中在点乘注意力的基础上又增加了1个缩放因子（ [公式] ）。增加的原因是：当 [公式] 较小时，两种注意力机制的表现情况类似，而 [公式] 增大时，点乘注意力的表现变差，认为是由于点乘后的值过大，导致softmax函数趋近于边缘，梯度较小；

- [搞清transformer结构](https://juejin.cn/post/6844903680487981069#comment)
