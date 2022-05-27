# Summary of lightgbm

[lightgbm paper](https://papers.nips.cc/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf)

[官方库lightgbm](https://github.com/microsoft/LightGBM)

[信息熵增益理解](https://www.zhihu.com/question/22104055)

- 熵:表示随机变量的不确定性。
- 条件熵：在一个条件下，随机变量的不确定性。
- 信息增益：熵 - 条件熵。表示在一个条件下，信息不确定性减少的程度。

#### **lightgbm 最大特点**
- ***通过使用leaf-wise分裂策略代替XGBoost的level-wise分裂策略***，通过只选择分裂增益最大的结点进行分裂，避免了某些结点增益较小带来的开销。
- ***另外LightGBM通过使用基于直方图的决策树算法，只保存特征离散化之后的值***，代替XGBoost使用exact算法中使用的预排序算法（预排序算法既要保存原始特征的值，也要保存这个值所处的顺序索引），减少了内存的使用，并加速的模型的训练速度。

#### **直方图算法基本实现**
- 是指先把连续的浮点特征值离散化成 k 个整数，同时构造一个宽度为 k 的直方图。在遍历数据的时候，根据离散化后的值作为是索引，在直方图中累积统计量，然后根据直方图的离散值，遍历寻找最优的分割点。在XGBoost中需要遍历所有离散化的值，而LightGBM通过建立直方图只需要遍历 k 个直方图的值。

![histogram.png](https://github.com/Yang-HangWA/DailyNote/blob/master/pic/histogram.png)

#### **使用直方图的优点**
- 明显减少内存的使用，因为直方图不需要额外存储预排序的结果，而且可以只保存特征离散化后的值。
    
- 遍历特征值时，不需要像XGBoost一样需要计算每次分裂的增益，而是对每个特征只需要计算建立直方图的个数，即k次，时间复杂度由O(#data * #feature)优化到O(k * #feature)。（由于决策树本身是弱分类器，分割点是否精确并不是太重要，因此直方图算法离散化的分割点对最终的精度影响并不大。另外直方图由于取的是较粗略的分割点，因此不至于过度拟合，起到了正则化的效果）

- LightGBM的直方图能做差加速。一个叶子的直方图可以由它的父亲结点的直方图与它兄弟的直方图做差得到。构造的直方图本来需要遍历该叶子结点上所有数据，但是直方图做差仅需遍历直方图的k个桶即可（即直方图区间），速度上可以提升一倍。如下图:

![split.png](https://github.com/Yang-HangWA/DailyNote/blob/master/pic/split.png)

#### **lightgbm 两大核心技术**
- （1）GOSS(Gradient-based One-Side Sampling)：减少样本数
- （2）EFB (Exclusive Feature Bundling ):减少特征数
    
- 小结: 直方图算法其实就是将value离散化了，生成一个个bin，常称为分桶，离散化后的bin数其实是小于原始的value数的，于是复杂度从（#feature*#data）降为了（#feature*#bin）。

[知乎佳作:【白话机器学习】算法理论+实战之LightGBM算法]( https://zhuanlan.zhihu.com/p/149522630)

#### **lightgbm优化思路来源**
- 寻找最优分裂点的复杂度 = 特征数量×分裂点的数量×样本的数量

所以如果想在xgboost上面做出一些优化的话，我们是不是就可以从上面的三个角度下手，比如想个办法减少点特征数量啊， 分裂点的数量啊， 样本的数量啊等等。 元芳，你怎么看？

哈哈， 微软里面提出lightgbm的那些大佬还真就是这样做的， Lightgbm里面的直方图算法就是为了减少分裂点的数量， Lightgbm里面的单边梯度抽样算法就是为了减少样本的数量， 而Lightgbm里面的互斥特征捆绑算法就是为了减少特征的数量。 并且后面两个是Lightgbm的亮点所在。

#### **GOSS ***&*** EFB**

GOSS的感觉就好像一个公寓里本来住了10个人，感觉太挤了，赶走了6个人，但是剩下的人要分摊他们6个人的房租。
单边梯度抽样算法基本上就理清楚了，Lightgbm正是通过这样的方式，在不降低太多精度的同时，减少了样本数量，使得训练速度加快。

单边梯度抽样算法(Gradient-based One-Side Sampling)是从减少样本的角度出发， 排除大部分权重小的样本，仅用剩下的样本计算信息增益，它是一种在减少数据和保证精度上平衡的算法。

GOSS 算法保留了梯度大的样本，并对梯度小的样本进行随机抽样，为了不改变样本的数据分布，在计算增益时为梯度小的样本引入一个常数进行平衡。首先将要进行分裂的特征的所有取值按照绝对值大小降序排序(xgboost也进行了排序，但是LightGBM不用保存排序后的结果），然后拿到前 的梯度大的样本，和剩下样本的，在计算增益时，后面的这通过乘上来放大梯度小的样本的权重。一方面算法将更多的注意力放在训练不足的样本上，另一方面通过乘上权重来防止采样对原始数据分布造成太大的影响。


比如，我们把特征A和B绑定到了同一个bundle里面， A特征的原始取值区间[0,10), B特征原始取值区间[0,20), 这样如果直接绑定，那么会发现我从bundle里面取出一个值5， 就分不出这个5到底是来自特征A还是特征B了。 所以我们可以再B特征的取值上加一个常量10转换为[10, 30)，这样绑定好的特征取值就是[0,30), 我如果再从bundle里面取出5， 就一下子知道这个是来自特征A的。 这样就可以放心的融合特征A和特征B了。


以简单的回忆一下，我们说xgboost在寻找最优分裂点的时间复杂度其实可以归到三个角度：特征的数量，分裂点的数量和样本的数量。 而LightGBM也提出了三种策略分别从这三个角度进行优化，直方图算法就是为了减少分裂点的数量， GOSS算法为了减少样本的数量，而EFB算法是为了减少特征的数量。


#### level-wise **vs** leaf-wise

 XGBoost 在树的生成过程中采用 Level-wise 的增长策略，该策略遍历一次数据可以同时分裂同一层的叶子，容易进行多线程优化，也好控制模型复杂度，不容易过拟合。但实际上Level-wise是一种低效的算法，因为它不加区分的对待同一层的叶子，实际上很多叶子的分裂增益较低，没必要进行搜索和分裂，因此带来了很多没必要的计算开销(一层一层的走，不管它效果到底好不好)

Leaf-wise 则是一种更为高效的策略，每次从当前所有叶子中，找到分裂增益最大的一个叶子，然后分裂，如此循环。因此同 Level-wise 相比，在分裂次数相同的情况下，Leaf-wise 可以降低更多的误差，得到更好的精度。Leaf-wise 的缺点是可能会长出比较深的决策树，产生过拟合。因此 LightGBM 在 Leaf-wise 之上增加了一个最大深度的限制，在保证高效率的同时防止过拟合。 （最大信息增益的优先， 我才不管层不层呢）

Level-wise的做法会产生一些低信息增益的节点，浪费运算资源，但是这个对于防止过拟合挺有用。 而Leaf-wise能够追求更好的精度，降低误差，但是会带来过拟合问题。 那你可能问，那为啥还要用Leaf-wise呢？ 过拟合这个问题挺严重鸭！ 但是人家能提高精度啊，哈哈，哪有那么十全十美的东西， 并且作者也使用了max_depth来控制树的高度。 其实敢用Leaf-wise还有一个原因就是Lightgbm在做数据合并，直方图和GOSS等各个操作的时候，其实都有天然正则化的作用，所以作者感觉在这里使用Leaf-wise追求高精度是一个不错的选择。


#### **GOSS原文图**
![GOSS.png](https://github.com/Yang-HangWA/DailyNote/blob/master/pic/GOSS.png)


#### **EFB原文图**
![EFB.png](https://github.com/Yang-HangWA/DailyNote/blob/master/pic/EFB.png)

#### 问题总结
- 【问】xgboost/gbdt在调参时为什么树的深度很少就能达到很高的精度？ 用xgboost/gbdt在在调参的时候把树的最大深度调成6就有很高的精度了。但是用DecisionTree/ RandomForest的时候需要把树的深度调到15或更高。用RandomForest所需要的树的深度和DecisionTree一样我能理解，因为它是用bagging的方法把DecisionTree组合在一起，相当于做了多次DecisionTree一样。但是xgboost/gbdt仅仅用梯度上升法就能用6个节点的深度达到很高的预测精度，使我惊讶到怀疑它是黑科技了。请问下xgboost/gbdt是怎么做到的？它的节点和一般的DecisionTree不同吗？  
- 这是一个非常好的问题，题主对各算法的学习非常细致透彻，问的问题也关系到这两个算法的本质。这个问题其实并不是一个很简单的问题，我尝试用我浅薄的机器学习知识对这个问题进行回答。 

- 一句话的解释，来自周志华老师的机器学习教科书（机器学习-周志华）：***Boosting主要关注降低偏差，因此Boosting能基于泛化性能相当弱的学习器构建出很强的集成；Bagging主要关注降低方差，因此它在不剪枝的决策树、神经网络等学习器上效用更为明显。*** 

- 随机森林(random forest)和GBDT都是属于集成学习（ensemble-learning)的范畴。集成学习下有两个重要的策略Bagging和Boosting。 

- Bagging算法是这样做的：每个分类器都随机从原样本中做有放回的采样，然后分别在这些采样后的样本上训练分类器，然后再把这些分类器组合起来。简单的多数投票一般就可以。其代表算法是随机森林。Boosting的意思是这样，他通过迭代地训练一系列的分类器，每个分类器采用的样本分布都和上一轮的学习结果有关。其代表算法是AdaBoost, GBDT。 
- 其实就机器学习算法来说，其泛化误差可以分解为两部分，偏差（bias)和方差(variance)。这个可由下图的式子导出（这里用到了概率论公式D(X)=E(X2)-[E(X)]2）。偏差指的是算法的期望预测与真实预测之间的偏差程度，反应了模型本身的拟合能力；方差度量了同等大小的训练集的变动导致学习性能的变化，刻画了数据扰动所导致的影响。这个有点儿绕，不过你一定知道过拟合。 
  如下图所示，当模型越复杂时，拟合的程度就越高，模型的训练偏差就越小。但此时如果换一组数据可能模型的变化就会很大，即模型的方差很大。所以模型过于复杂的时候会导致过拟合。 
  当模型越简单时，即使我们再换一组数据，最后得出的学习器和之前的学习器的差别就不那么大，模型的方差很小。还是因为模型简单，所以偏差会很大。 

![bias&varinace](https://github.com/Yang-HangWA/DailyNote/blob/master/pic/bias_and_variance.png)

#### **模型复杂度与偏差方差的关系图**
- 也就是说，当我们训练一个模型时，偏差和方差都得照顾到，漏掉一个都不行。 
- 对于Bagging算法来说，由于我们会并行地训练很多不同的分类器的目的就是降低这个方差(variance) ,因为采用了相互独立的基分类器多了以后，h的值自然就会靠近.所以对于每个基分类器来说，目标就是如何降低这个偏差（bias),所以我们会采用深度很深甚至不剪枝的决策树。 
- 对于Boosting来说，每一步我们都会在上一轮的基础上更加拟合原数据，所以可以保证偏差（bias）,所以对于每个基分类器来说，问题就在于如何选择variance更小的分类器，即更简单的分类器，所以我们选择了深度很浅的决策树。
- 最近引起关注的一个Gradient-Boosting算法：xgboost，在计算速度和准确率上，较GBDT有明显的提升。xgboost的全称是eXtreme Gradient Boosting，它是Gradient Boosting Machine的一个c++实现，作者为正在华盛顿大学研究机器学习的大牛陈天奇 。xgboost最大的特点在于，它能够自动利用CPU的多线程进行并行，同时在算法上加以改进提高了精度。它的处女秀是Kaggle的希格斯子信号识别竞赛，因为出众的效率与较高的预测准确度在比赛论坛中引起了参赛选手的广泛关注。值得我们在GBDT的基础上对其进一步探索学习

### **实践相关**

#### **lightgbm predict(python）用法踩坑**

当我们用lightgbm训练好一个模型，测试也有不错的效果，打算投入生产的时候，却发现lightgbm部署起来并没有那么友善。
如果我们每次用于预测的数据只有一条时，predict的效率其实是很低的，主要原因有两点：
   
   - 1.predict每次调用会初始化一个predictor的类，具体可参见issue: https://github.com/microsoft/LightGBM/issues/906
   - 2.predict调用时默认使用多线程，并且会使用掉所有剩余cpu。单条串行无疑会极大降低predict的处理速度。

当然，如果我们的应用场景需要单条处理，也是可以勉强用的，但是记住一定要设置num_threads=1,不然会发现服务器cpu资源会被吃完，如果正好这台服务器还有其他服务，可能会受到影响。具体设置方法，参见issue: https://github.com/microsoft/LightGBM/issues/1534

但是lightgbm社区维护者也建议每次尽量多输入多条预测，否则效率真的很低。具体可参见讨论： https://github.com/microsoft/LightGBM/issues/906


#### **利用treelite包部署lightgbm模型**
除了lightgbm自带的predict函数用于预测之外，我们可以利用项目treelite对模型进行转换，项目地址： https://github.com/dmlc/treelite
    项目文档： https://treelite.readthedocs.io/en/latest/
    
- 1.treelite安装(用编译安装最新版，如果用pip命令安装可能无法使用到最新的版本，旧版本中功能正确性上更低)
安装命令如下：
    - git到本地
    ```
    git clone --recursive https://github.com/dmlc/treelite.git
    cd treelite
    ```
    - 编译安装
    ```
    mkdir build
    cd build
    cmake ..
    cd ..
    ```
    - 安装python包
    ```
    cd python
    sudo python setup.py install
    ```   
- 2.例子说明
    - treelite的用法可以参考文档，也可以参考下面这个例子:
        
    - eg. 对于已经训练好的模型 : value_model.txt 进行部署模型转化
    ```
    #load中的为元模型，model_format设置模型类型，有lightgbm,xgboost等  
    #model.epxort_lib导出*.so模型；libpath为导出路径；toolchain为编译器类型，不同系统使用的类型不同；params用于设置参数，parallel_comp设置可以加快导出速度
    model = treelite.Model.load('value_model.txt', model_format='lightgbm')
    model.export_lib(toolchain='gcc', libpath='./treelite/big_value_model.so', params={'parallel_comp':32}, verbose=True)
    ```
    但是目前这个项目版本为0.32，还有些bug, 如果使用，一定要对原始模型和转化之后的模型的计算结果进行对比，看是否相同。   

3.建议方式
- 预加载模型进行预测。


#### 贝叶斯相关
- [lightGBM区间估计 issue](https://github.com/Microsoft/LightGBM/issues/1036)

- [Quantiles Regression is Poorly-Calibrated](https://github.com/Microsoft/LightGBM/issues/1182)

- [Simple Bayesian Optimization for LightGBM(贝叶斯超参数优化)](https://www.kaggle.com/sz8416/simple-bayesian-optimization-for-lightgbm)

- https://blog.csdn.net/ssswill/article/details/86564056

- https://lightgbm.readthedocs.io/en/latest/Experiments.html

- https://github.com/Microsoft/LightGBM/blob/master/examples/python-guide/simple_example.py

- https://www.kaggle.com/fabiendaniel/hyperparameter-tuning/notebook

- https://www.kaggle.com/rsmits/bayesian-optimization-for-lightgbm-parameters/notebook


- [贝叶斯理论部分解释（cnblog）](https://www.cnblogs.com/yangruiGB2312/p/9374377.html)

- [贝叶斯优化github](https://github.com/fmfn/BayesianOptimization/blob/master/examples/basic-tour.ipynb)


#### lightgbm GPU配置
- 1.cmake安装：https://cmake.org/files/v3.12/

```
wget   https://cmake.org/files/v3.12/cmake-3.12.0.tar.gz
cd cmake
./configure
make && make install 
```

- 2.opencl报错处理
```
apt update
apt install ocl-icd-opencl-dev
```

- 3.最后安装命令：
```
pip install lightgbm --install-option=--gpu --install-option="--opencl-include-dir=/usr/local/cuda/include/" --install-option="--opencl-library=/usr/local/cuda/lib64/libOpenCL.so"
```

- [最简便的lightGBM GPU支持的安装、验证方法](https://blog.csdn.net/lccever/article/details/80535058)


#### How to input  categorical features 

- [python报错：使用lgb过程中报错：DataFrame.dtypes for data must be int, float or bool](https://blog.csdn.net/qq_27877617/article/details/83653129)

- [convert them to int first](https://github.com/microsoft/LightGBM/issues/345#issuecomment-286081942)

- [categorical feature use(project of self help training)](https://github.com/microsoft/LightGBM/issues/1408)

- [correct way to use categorical feature](https://github.com/microsoft/LightGBM/issues/2695)

- **example**
```
train = self.__lgb.Dataset(
data=train_feature,
label=train_label,
feature_name=train_feature.columns.tolist(),
categorical_feature=self.__categorical_feature.tolist()
)
```
