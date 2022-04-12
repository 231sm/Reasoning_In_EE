[**中文**](https://github.com/231sm/Reasoning_In_EE/blob/main/baselines/README.md) | [**English**](https://github.com/231sm/Reasoning_In_EE/blob/main/baselines/README_EN.md)

# 基线模型

这个文件夹下包含论文中使用的基线模型和经过处理的论文数据集，并且可以在论文中的数据集上复现（***包含实验结果和训练好的模型权重下载链接***）。经过微调超参数和训练更长时间，当前模型得以在论文的数据集中有更好的基准表现。

## 包含模型

- [DMCNN](https://github.com/231sm/Reasoning_In_EE/blob/main/baselines/DMCNN/README.md)
- [JMEE](https://github.com/231sm/Reasoning_In_EE/blob/main/baselines/JMEE/README.md)
- [JRNN](https://github.com/231sm/Reasoning_In_EE/blob/main/baselines/JRNN/README.md)

## 关于事件检测任务中P，R，F1值的计算方式讨论

**本项目使用的评估指标采用清华大学[MAVEN](https://github.com/THU-KEG/MAVEN-dataset)的计算方式，使用多分类方式建模问题，所以调用`sklearn.metrics`相关方法进行指标计算。**

![公式截图](https://github.com/231sm/Reasoning_In_EE/tree/main/baselines/equations.png)

### 实际案例说明

以3分类，10个样本为例，假设样本标签为`y=[0,0,0,1,1,1,1,2,2,2]`，运行如下代码：

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

可以找出满足要求的几组预测`pred`：

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
