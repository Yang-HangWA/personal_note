# Summary of gnn

#### 知识图谱
- [这是一份通俗易懂的知识图谱技术与应用指南](https://www.jiqizhixin.com/articles/2018-06-20-4)

- [GNN系列](https://zhuanlan.zhihu.com/p/76175953)


### dgl 

* dockerfile in https://github.com/dmlc/dgl/tree/master/docker
* install 教程：https://docs.dgl.ai/install/index.html#install-from-source

* 实际操作：
使用dockerfile : Dockerfile.ci_gpu

* 并在容器中执行：
`conda install -c dglteam dgl-cuda9.0`

dockerfile中cuda版本要与命令中的cuda版本保持一致。

### 知识抽取
- 信息来源：传统数据库+网页爬取信息
- 难点：信息抽取的难点在于处理非结构化数据。

- 主要涉及以下几个方面的自然语言处理技术：  

    - a. 实体命名识别（Name Entity Recognition）    

    - b. 关系抽取（Relation Extraction）    

    - c. 实体统一（Entity Resolution）    

    - d. 指代消解（Coreference Resolution）

- 在实体命名识别和关系抽取过程中，有两个比较棘手的问题：一个是实体统一，也就是说有些实体写法上不一样，但其实是指向同一个实体。比如“NYC”和“New York”表面上是不同的字符串，但其实指的都是纽约这个城市，需要合并。实体统一不仅可以减少实体的种类，也可以降低图谱的稀疏性（Sparsity）；另一个问题是指代消解，也是文本中出现的“it”, “he”, “she”这些词到底指向哪个实体，比如在本文里两个被标记出来的“it”都指向“hotel”这个实体。

- 知识图谱主要有两种存储方式：一种是基于RDF的存储；另一种是基于图数据库的存储。

### NLP详解文章
- [【完结】 12篇文章带你完全进入NLP领域，掌握核心技术](https://zhuanlan.zhihu.com/p/80217404)

### HMM与CRF
- [CRF](https://link.zhihu.com/?target=https%3A//mp.weixin.qq.com/s%3F__biz%3DMzA3NDIyMjM1NA%3D%3D%26mid%3D2649033930%26idx%3D2%26sn%3D19041a898ee193d215d2353a6ab3eb6e%26chksm%3D8712b2b7b0653ba1f27f9288e8642d120ab850bf9cfae5293af7b027d108c1901c3e2d123629%26scene%3D21%23wechat_redirect)

- 条件随机场(CRF)在现今NLP中序列标记任务中是不可或缺的存在。太多的实现基于此，例如LSTM+CRF，CNN+CRF，BERT+CRF。因此，这是一个必须要深入理解和吃透的模型。

- [RNN](https://mp.weixin.qq.com/s?__biz=MzA3NDIyMjM1NA==&mid=2649034382&idx=2&sn=6d2b2efc32eeb861fc58a1c8a612eb9d&chksm=8712b0f3b06539e5afe386786b291e12296843ee5b0ba0300d79807efb1e14ce9f66cb544400&scene=21#wechat_redirect)

- [分词隐马尔可夫](https://mp.weixin.qq.com/s?__biz=MzA3NDIyMjM1NA==&mid=2649033804&idx=2&sn=faf79533669416849e807a0aeb2c7eee&chksm=8712b231b0653b27f94018743d6f4f64582a2a29d3caa01d61e71cfaf89a8d32dbcb877e105e&scene=21#wechat_redirect)
- 维特比算法是计算一个概率最大的路径，如图要计算“我爱中国”的分词序列

- [bert](https://zhuanlan.zhihu.com/p/46652512)
- BERT的全称是Bidirectional Encoder Representation from Transformers，即双向Transformer的Encoder，因为decoder是不能获要预测的信息的。模型的主要创新点都在pre-train方法上，即用了Masked LM和Next Sentence Prediction两种方法分别捕捉词语和句子级别的representation。

- [book:Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/ed3book.pdf)