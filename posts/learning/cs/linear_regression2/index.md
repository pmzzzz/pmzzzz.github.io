# 线性回归(2)--从零实现


使用基础的`Tensor`以及`autograd`实现线性回归模型的构建和训练
<!--more-->
{{<admonition tip "注意">}}
本文主要记录代码，优化细节，添加注释。
{{</admonition>}}
## 细节highlight

1. 样本数量不能被batch_size整除时的处理
1. 用`matmul`自适应矩阵的向量的乘法
1. 误差函数中的y需要reshape，保证维度一致
1. 在`with torch.no_grad():`中更新参数
1. 参数更新后需要`.grad.zero_()`将梯度置零

``` python 
%matplotlib inline
import torch
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import random
```


## 生成数据集

我们构造一个简单的人工训练数据集，它可以使我们能够直观比较学到的参数和真实的模型参数的区别。设训练数据集样本数为1000，输入个数（特征数）为2。给定随机生成的批量样本特征 $\boldsymbol{X} \in \mathbb{R}^{1000 \times 2}$，我们使用线性回归模型真实权重 $\boldsymbol{w} = [2, -3.4]^\top$ 和偏差 $b = 4.2$，以及一个随机噪声项 $\epsilon$ 来生成标签
$$
\boldsymbol{y} = \boldsymbol{X}\boldsymbol{w} + b + \epsilon
$$

其中噪声项 $\epsilon$ 服从均值为0、标准差为0.01的正态分布。噪声代表了数据集中无意义的干扰。下面，让我们生成数据集。

``` python
num_inputs = 2 # 输入维度
num_examples = 1000 # 样本数量
true_w = [2, -3.4] 
true_b = 4.2
features = torch.randn(num_examples, num_inputs,
                       dtype=torch.float32) # 随机生成的样本
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b # 计算精确的标签
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()),
                       dtype=torch.float32) # 给标签加上随机干扰
```

注意，`features`的每一行是一个长度为2的向量，而`labels`的每一行是一个长度为1的向量（标量）。

``` python
print(features[0], labels[0])
```
输出：
```
tensor([0.8557, 0.4793]) tensor(4.2887)
```

通过生成第二个特征`features[:, 1]`和标签 `labels` 的散点图，可以更直观地观察两者间的线性关系。

``` python
def use_svg_display():
    # 用矢量图显示
    display.set_matplotlib_formats('svg')

def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    # 设置图的尺寸
    plt.rcParams['figure.figsize'] = figsize

set_figsize()
plt.scatter(features[:, 1].numpy(), labels.numpy(), 1);
```
{{<image src="output1.png" caption="">}}

## 读取数据

在训练模型的时候，我们需要遍历数据集并不断读取小批量数据样本。这里我们定义一个函数：它每次返回`batch_size`（批量大小）个随机样本的特征和标签。

``` python
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # 这些样本是随机读取的，没有特定的顺序
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)]) # i 每次增加一个batch_size,最后一次可能超过下标，所以用min()保证不超过
        yield features[batch_indices], labels[batch_indices]
```

让我们读取第一个小批量数据样本并打印。每个批量的特征形状为(10, 2)，分别对应批量大小和输入个数；标签形状为批量大小。
``` python
batch_size = 10

for X, y in data_iter(batch_size, features, labels):
    print(X, y)
    break
```
输出：
```
tensor([[-1.4239, -1.3788],
        [ 0.0275,  1.3550],
        [ 0.7616, -1.1384],
        [ 0.2967, -0.1162],
        [ 0.0822,  2.0826],
        [-0.6343, -0.7222],
        [ 0.4282,  0.0235],
        [ 1.4056,  0.3506],
        [-0.6496, -0.5202],
        [-0.3969, -0.9951]]) 
 tensor([ 6.0394, -0.3365,  9.5882,  5.1810, -2.7355,  5.3873,  4.9827,  5.7962,
         4.6727,  6.7921])
```

## 初始化模型参数

我们将权重初始化成均值为0、标准差为0.01的正态随机数，偏差则初始化成0。

``` python
w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)), dtype=torch.float32)
b = torch.zeros(1, dtype=torch.float32)
```

之后的模型训练中，需要对这些参数求梯度来迭代参数的值，因此我们要让它们的`requires_grad=True`。
``` python
w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True) 
```

## 定义模型

下面是线性回归的矢量计算表达式的实现。我们使用`mm`函数做矩阵乘法。

``` python
def linreg(X, w, b):
    return torch.matmul(X, w) + b # 使用matmul自动应对更多情况
```

## 定义损失函数

我们使用上一节描述的平方损失来定义线性回归的损失函数。在实现中，我们需要把真实值`y`变形成预测值`y_hat`的形状。以下函数返回的结果也将和`y_hat`的形状相同。

``` python
def squared_loss(y_hat, y):  # 本函数已保存在d2lzh_pytorch包中方便以后使用
    # 注意这里返回的是向量, 另外, pytorch里的MSELoss并没有除以 2
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2 # 注意需要reshape
```

## 定义优化算法

以下的`sgd`函数实现了上一节中介绍的小批量随机梯度下降算法。它通过不断迭代模型参数来优化损失函数。这里自动求梯度模块计算得来的梯度是一个批量样本的梯度和。我们将它除以批量大小来得到平均值。

``` python
def sgd(params, lr, batch_size):  #@save
    """小批量随机梯度下降"""
    with torch.no_grad(): # 更新参数时不更新梯度
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_() # 梯度清零
```

## 训练模型

在训练中，我们将多次迭代模型参数。在每次迭代中，我们根据当前读取的小批量数据样本（特征`X`和标签`y`），通过调用反向函数`backward`计算小批量随机梯度，并调用优化算法`sgd`迭代模型参数。由于我们之前设批量大小`batch_size`为10，每个小批量的损失`l`的形状为(10, 1)。回忆一下自动求梯度一节。由于变量`l`并不是一个标量，所以我们可以调用`.sum()`将其求和得到一个标量，再运行`l.backward()`得到该变量有关模型参数的梯度。注意在每次更新完参数后不要忘了将参数的梯度清零。

在一个迭代周期（epoch）中，我们将完整遍历一遍`data_iter`函数，并对训练数据集中所有样本都使用一次（假设样本数能够被批量大小整除）。这里的迭代周期个数`num_epochs`和学习率`lr`都是超参数，分别设3和0.03。在实践中，大多超参数都需要通过反复试错来不断调节。虽然迭代周期数设得越大模型可能越有效，但是训练时间可能过长。而有关学习率对模型的影响，我们会在后面“优化算法”一章中详细介绍。

``` python
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)  # X和y的小批量损失
        # 因为l形状是(batch_size,1)，而不是一个标量。l中的所有元素被加到一起，
        # 并以此计算关于[w,b]的梯度
        l.sum().backward()
        sgd([w, b], lr, batch_size)  # 使用参数的梯度更新参数
    with torch.no_grad(): # 计算loss的时候不更新梯度
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
```
输出：
```
epoch 1, loss 0.028127
epoch 2, loss 0.000095
epoch 3, loss 0.000050
```

训练完成后，我们可以比较学到的参数和用来生成训练集的真实参数。它们应该很接近。

``` python
print(true_w, '\n', w)
print(true_b, '\n', b)
```
输出：
```
[2, -3.4] 
 tensor([[ 1.9998],
        [-3.3998]], requires_grad=True)
4.2 
 tensor([4.2001], requires_grad=True)
```

## 小结

* 可以看出，仅使用`Tensor`和`autograd`模块就可以很容易地实现一个模型。
