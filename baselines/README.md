[**中文**](https://github.com/231sm/Reasoning_In_EE/blob/main/baselines/README.md) | [**English**](https://github.com/231sm/Reasoning_In_EE/blob/main/baselines/README_EN.md)

# 基线模型

这个文件夹下包含论文中使用的基线模型和经过处理的论文数据集，并且可以在论文中的数据集上复现（***包含实验结果和训练好的模型权重下载链接***）。经过微调超参数和训练更长时间，当前模型得以在论文的数据集中有更好的基准表现。

## 包含模型

- [DMCNN](https://github.com/231sm/Reasoning_In_EE/blob/main/baselines/DMCNN/README.md)
- [JMEE](https://github.com/231sm/Reasoning_In_EE/blob/main/baselines/JMEE/README.md)
- [JRNN](https://github.com/231sm/Reasoning_In_EE/blob/main/baselines/JRNN/README.md)

## 关于事件检测任务中P，R，F1值的计算方式讨论

**本项目使用的评估指标采用清华大学[MAVEN](https://github.com/THU-KEG/MAVEN-dataset)的计算方式，使用多分类方式建模问题，所以调用`sklearn.metrics`相关方法进行指标计算。**

由于多分类任务下micro指标Precision、Recall和F1是相等的（因为多分类的micro计算中FN=FP），本论文中采用的是macro的Precision和Recall计算方式。同时，与其他事件抽取任务保持一致地，本文采用micro方式计算F1 score指标。

下面对论文中的情况，即$\rm F1_{micro}\gt P_{macro},R_{macro}$的情形进行说明。

假设对一个$N$分类任务，设样本总量为$M$，预测的混淆矩阵为$C$，其中$C_{ij}$表示标签为第$i$类，预测为第$j$类的样本数，$1\leq i,j\le N$。

于是：

$\rm P_{macro}=\frac{1}{N}\sum_{i=1}^{N}P_{macro,i}=\frac{1}{N}\sum_{i=1}^{N}\frac{{TP}_i}{{TP}_i+{FP}_i}$

$\rm =\frac{1}{N}\sum_{i=1}^{N}\frac{C_{ii}}{C_{ii}+\sum_{j\ne i}{C_{ji}}}=\frac{1}{N}\sum_{i=1}^{N}\frac{C_{ii}}{\sum_{j=1}^{N}C_{ji}}.$

$\rm R_{macro}=\frac{1}{N}\sum_{i=1}^{N}R_{macro,i}=\frac{1}{N}\sum_{i=1}^{N}\frac{{TP}_i}{{TP}_i+{FN}_i}$

$\rm =\frac{1}{N}\sum_{i=1}^{N}\frac{C_{ii}}{C_{ii}+\sum_{j\ne i}C_{ij}}=\frac{1}{N}\sum_{i=1}^{N}\frac{C_{ii}}{\sum_{j=1}^{N}C_{ij}}.$

$\rm F1_{micro}=\frac{TP}{TP+FN}=\frac{\sum_{i=1}^{N}{C_{ii}}}{\sum_{i=1}^{N}{C_{ii}}+\sum_{i\ne j}C_{ij}}=\frac{\sum_{i=1}^{N}C_{ii}}{M}.$

不妨从$\rm F1_{micro}\gt R_{macro}$开始说明。

注意到$\rm M=\sum_{i=1}^{N}\sum_{j=1}^{N}C_{ij}$，于是有$\rm \frac{\sum_{i=1}^{N}C_{ii}}{\sum_{i=1}^{N}(\sum_{j=1}^{N}C_{ij})}\gt\frac{1}{N}\sum_{i=1}^{N}\frac{C_{ii}}{\sum_{j=1}^{N}C_{ij}}$

令$\rm a_i=C_{ii},b_i=\sum_{j}C_{ij}(0\le a_i\le b_i,1\le i\le N)$即第$i$类样本的数量，那么原命题等价于：$\rm \frac{\sum_{i}{a_i}}{\sum_{i}b_i}\gt \frac{1}{N}\sum_{i}\frac{a_i}{b_i}.$

若令$\rm b_i=\sum_j C_{ji}$即预测为第$i$类的样本数量，那么这一不等式也可表达$\rm F1_{micro}\gt P_{macro}$的情况。

下面要说明，在多分类任务中，**类别数量不均衡且分类表现不均衡的情况**会导致micro F1比macro P、R高。

首先讨论两种简单的情形：

- 不同类别的样本数量一致时，即$\rm b_i=B$，那么可知左右两侧计算结果相等；
- 不同类别的分类效果一致时，即$\rm \frac{a_j}{b_j}=\frac{b_i}{a_i},i\ne j$，那么可知左右两侧计算结果也是相等的。

而在实际情况中，由于一些数据集的长尾性质，即存在部分类别的样本数量$\rm b_i$较小，而模型表现也受其数量影响（同时也受该样本难度影响），导致分类的P/R情况即$\rm \frac{a_i}{b_i}$较小，即$\rm a_i, b_i\sim0, b_i\gg a_i$。那么有：

- 左侧micro计算总体情况需要累加$\rm \sum_i a_i,\sum_i b_i$，由于$\rm a_i,b_i\sim0$，因此这些类别对计算结果影响较小；
- 右侧macro方式计算$\rm\sum_i\frac{a_i}{b_i}$，由于$\rm \frac{a_i}{b_i}\sim0$，总体均值受其影响较大，导致数值偏低。

综合两方面考虑，右侧比左侧结果偏低，这是合理的结果。对于P值的讨论也属于类似的情况。

### 实际案例说明

以3分类，10个样本为例，假设样本标签为$\rm y=[0,0,0,1,1,1,1,2,2,2]$，运行如下代码：

```python
from random import randint
from sklearn.metrics import f1_score, precision_score, recall_score

num_labels = 3
y = [0, 0, 0, 1, 1, 1, 1, 2, 2, 2]
while True:
    pred = [randint(0, 2) for _ in range(len(y))]
    if len(set(pred)) < num_labels:
        continue
    pi = precision_score(y, pred, average='micro')
    pa = precision_score(y, pred, average='macro')
    ri = recall_score(y, pred, average='micro')
    ra = recall_score(y, pred, average='macro')
    fi = f1_score(y, pred, average='micro')
    fa = f1_score(y, pred, average='macro')
    if fi > pa and fi > ra:
        print('pred =', pred)
        print('micro: p = {:.4f}, r = {:.4f}, f1 = {:.4f}'.format(pi, ri, fi))
        print('macro: p = {:.4f}, r = {:.4f}, f1 = {:.4f}'.format(pa, ra, fa))
```

可以找出满足要求的几组预测$\rm pred$：

```python
# Case 0:
pred = [1, 2, 0, 1, 1, 1, 1, 2, 2, 0]
micro: p = 0.7000, r = 0.7000, f1 = 0.7000
macro: p = 0.6556, r = 0.6667, f1 = 0.6519

# Case 1:
pred = [0, 2, 1, 2, 1, 1, 1, 2, 2, 0]
micro: p = 0.6000, r = 0.6000, f1 = 0.6000
macro: p = 0.5833, r = 0.5833, f1 = 0.5738

# Case 2:
pred = [1, 2, 1, 1, 0, 1, 0, 2, 2, 1]
micro: p = 0.4000, r = 0.4000, f1 = 0.4000
macro: p = 0.3556, r = 0.3889, f1 = 0.3704

# Case 3:
pred = [1, 2, 2, 0, 1, 2, 2, 1, 1, 2]
micro: p = 0.2000, r = 0.2000, f1 = 0.2000
macro: p = 0.1500, r = 0.1944, f1 = 0.1667

# ...
```
