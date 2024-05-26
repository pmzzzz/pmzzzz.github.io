# softmax(3)--简洁实现



{{<admonition tip "注意">}}
本文主要记录代码，优化细节，添加注释。
{{</admonition>}}

## 加载数据
```python
import torch
from torch import nn
from d2l import torch as d2l
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
```
## 初始化参数
```python
# PyTorch不会隐式地调整输入的形状。因此，
# 我们在线性层前定义了展平层（flatten），来调整网络输入的形状
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights);
```

## 损失函数
```python
loss = nn.CrossEntropyLoss(reduction='none')
```
> 这里的交叉熵包括了softmax,也就是说，这一层直接接受**原始输出**就可以

## 优化器

```python
trainer = torch.optim.SGD(net.parameters(), lr=0.1)

```

## 训练

```python
num_epochs = 10
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```
