# 读写文件

## 加载和保存张量
```python
import torch
from torch import nn
from torch.nn import functional as F

x = torch.arange(4)
torch.save(x, 'x-file')
x2 = torch.load('x-file')

```
> 使用torch.save 和torch.load 来保存和读取张量，也可保存列表和字典
```python
y = torch.zeros(4)
torch.save([x, y],'x-files')
mydict = {'x': x, 'y': y}
torch.save(mydict, 'mydict')
```

## 加载和保存模型参数

```python
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))

net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)
torch.save(net.state_dict(), 'mlp.params')

```
使用保存net的stat_dict()
```python
clone = MLP()
clone.load_state_dict(torch.load('mlp.params'))
clone.eval()
```
使用load_state_dict加载数据，然后调用eval()

## 小结

* `save`和`load`函数可用于张量对象的文件读写。
* 我们可以通过参数字典保存和加载网络的全部参数。
* 保存架构必须在代码中完成，而不是在参数中完成。
