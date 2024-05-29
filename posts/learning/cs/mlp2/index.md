# MLP(2)--从零实现

## 数据导入
```python
import torch
from torch import nn
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size) # 每次随机取batch_size个样本
```

## 初始化模型参数
```python

num_inputs, num_outputs, num_hiddens = 784, 10, 256
W1 = nn.Parameter(torch.randn(
    num_inputs, num_hiddens, requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
W2 = nn.Parameter(torch.randn(
    num_hiddens, num_outputs, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))

params = [W1, b1, W2, b2]
```
> W初始化方差为0.01

## 激活函数
```python
def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)
```
> 这样写不在乎X的形状
## 模型
```python
def net(X):
    X = X.reshape((-1, num_inputs))
    H = relu(X@W1 + b1)  # 这里“@”代表矩阵乘法
    return (H@W2 + b2)
```
> 这里reshape保证特征的数量为784个
## 损失函数

```python
loss = nn.CrossEntropyLoss(reduction='none')
```
## 训练

```python
num_epochs, lr = 10, 0.1
updater = torch.optim.SGD(params, lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
```
> 使用了softmax章节的训练方法[softmax训练方法代码详解]()
