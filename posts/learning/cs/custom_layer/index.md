# 自定义层


## 不带参数的层
```python
import torch
import torch.nn.functional as F
from torch import nn


class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()
```

## 带参数的层
```python
class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units,))
    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)
```
注意使用`nn.Parameters`定义参数，此时可以使用`named_parameters()`查看参数
```python
paras = linear.named_parameters()
print(next(zzz))
print(next(zzz))
```
