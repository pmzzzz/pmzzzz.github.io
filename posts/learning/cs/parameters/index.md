# 参数管理


## 参数访问
```python
import torch
from torch import nn

net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
X = torch.rand(size=(2, 4))
print(net[2].state_dict())
```
```
OrderedDict([('weight', tensor([[-0.0427, -0.2939, -0.1894,  0.0220, -0.1709, -0.1522, -0.0334, -0.2263]])), ('bias', tensor([0.0887]))])
```
使用`state_dict()`方法返回一个模型的参数设置，在这个例子中，我们查看的`nn.Linear(8, 1)`的参数，由weight 和 bias组成
### 目标参数
```python
print(type(net[2].bias))
print(net[2].bias)
print(net[2].bias.data)
```

参数是复合的对象，包含值、梯度和额外信息。
这就是我们需要显式参数值的原因。
除了值之外，我们还可以访问每个参数的梯度。
在上面这个网络中，由于我们还没有调用反向传播，所以参数的梯度处于初始状态。
```python
net[2].weight.grad == None
# true
```
### 访问所有参数
```python
print(*[(name, param.shape) for name, param in net[0].named_parameters()])
print(*[(name, param.shape) for name, param in net.named_parameters()])
```
使用`named_parameters()`方法访问所有参数，返回名称以及参数本身

## 参数初始化
### 内置初始化
```python
nn.init.normal_(m.weight, mean=0, std=0.01)
nn.init.zeros_(m.bias)
nn.init.constant_(m.weight, 1)
nn.init.xavier_uniform_(m.weight)
```
以上时常用的内置初始化方法：正态、1、常数、xavier
```python
def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)
net.apply(init_constant)
net[0].weight.data[0], net[0].bias.data[0]
```
具体应用方法可以使用`.apply()`方法传入函数来控制初始化方式
### 自定义初始化
有时，深度学习框架没有提供我们需要的初始化方法。
在下面的例子中，我们使用以下的分布为任意权重参数$w$定义初始化方法：

$$
\begin{aligned}
    w \sim \begin{cases}
        U(5, 10) & \text{ 可能性 } \frac{1}{4} \\\
            0    & \text{ 可能性 } \frac{1}{2} \\\
        U(-10, -5) & \text{ 可能性 } \frac{1}{4}
    \end{cases}
\end{aligned}
$$
```python
def my_init(m):
    if type(m) == nn.Linear:
        print("Init", *[(name, param.shape)
                        for name, param in m.named_parameters()][0])
        nn.init.uniform_(m.weight, -10, 10)
        m.weight.data *= m.weight.data.abs() >= 5

net.apply(my_init)
net[0].weight[:2]
```

## 参数绑定
有时我们希望在多个层间共享参数：
我们可以定义一个稠密层，然后使用它的参数来设置另一个层的参数。
```python
# 我们需要给共享层一个名称，以便可以引用它的参数
shared = nn.Linear(8, 8)
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                    shared, nn.ReLU(),
                    shared, nn.ReLU(),
                    nn.Linear(8, 1))
net(X)
# 检查参数是否相同
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0, 0] = 100
# 确保它们实际上是同一个对象，而不只是有相同的值
print(net[2].weight.data[0] == net[4].weight.data[0])
```
这个例子表明第三个和第五个神经网络层的参数是绑定的。
它们不仅值相等，而且由相同的张量表示。
因此，如果我们改变其中一个参数，另一个参数也会改变。
这里有一个问题：当参数绑定时，梯度会发生什么情况？
答案是由于模型参数包含梯度，因此在反向传播期间第二个隐藏层
（即第三个神经网络层）和第三个隐藏层（即第五个神经网络层）的梯度会加在一起。
