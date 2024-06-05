# 模型构建

## 层和块
层和块本质没有区别，都可以看作是一个黑盒子，有输入和输出，都可以当作单独的模块进行组合，组合之后的产物又是一个新的块。就像一本书一样，分为几个大的章节，章节里面又有很多知识点，知识点下面有很多段落。对我们而言，不管什么颗粒度，它们都是由文字组成的知识。

## 自定义块
如前面所说，一个块可以拆解成如下几个要素：

1. 将输入数据作为其前向传播函数的参数。
1. 通过前向传播函数来生成输出。请注意，输出的形状可能与输入的形状不同。例如，我们上面模型中的第一个全连接的层接收一个20维的输入，但是返回一个维度为256的输出。
1. 计算其输出关于输入的梯度，可通过其反向传播函数进行访问。通常这是自动发生的。
1. 存储和访问前向传播计算所需的参数。
1. 根据需要初始化模型参数。

```python
class MLP(nn.Module):
    # 用模型参数声明层。这里，我们声明两个全连接的层
    def __init__(self):
        # 调用MLP的父类Module的构造函数来执行必要的初始化。
        # 这样，在类实例化时也可以指定其他函数参数，例如模型参数params（稍后将介绍）
        super().__init__()
        self.hidden = nn.Linear(20, 256)  # 隐藏层
        self.out = nn.Linear(256, 10)  # 输出层

    # 定义模型的前向传播，即如何根据输入X返回所需的模型输出
    def forward(self, X):
        # 注意，这里我们使用ReLU的函数版本，其在nn.functional模块中定义。
        return self.out(F.relu(self.hidden(X)))
```
上述代码构造了一个MLP模块，它内部由两个全连接层组成，并且规定了前向传播的计算方法，这就是一个自定义的块。


## 顺序块
顺序块是一个基础常用的块的结构，它的构建方式就是将n个块或者层顺序的链接起来，甚至不需要写forward函数
```python
class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            # 这里，module是Module子类的一个实例。我们把它保存在'Module'类的成员
            # 变量_modules中。_module的类型是OrderedDict
            self._modules[str(idx)] = module

    def forward(self, X):
        # OrderedDict保证了按照成员添加的顺序遍历它们
        for block in self._modules.values():
            X = block(X)
        return X
```
以上代码展示了顺序块内部执行的逻辑：
1. 在初始化的时候顺序将块存入`_modules`中
2. 前向传播则是按照添加顺序依次进行运算
```python
net = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
```
以上代码展示了`MySequential`类的调用方法，与`nn.Sequential`并无差异

## 在前向传播函数中执行代码

在我们构建自己的模型时，需要进行python的流程控制【例如将某一层的结果放大两倍】，这个时候顺序块就不满足我们需求了。需要自定义前向传播的方法

```python
class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # 不计算梯度的随机权重参数。因此其在训练期间保持不变
        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        self.linear = nn.Linear(20, 20)

    def forward(self, X):
        X = self.linear(X)
        # 使用创建的常量参数以及relu和mm函数
        X = F.relu(torch.mm(X, self.rand_weight) + 1)
        # 复用全连接层。这相当于两个全连接层共享参数
        X = self.linear(X)
        # 控制流
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()
```
在上面这个例子当中在这个`FixedHiddenMLP`模型中，我们实现了一个隐藏层，
其权重（`self.rand_weight`）在实例化时被随机初始化，之后为常量。
这个权重不是一个模型参数，因此它永远不会被反向传播更新。
然后，神经网络将这个固定层的输出通过一个全连接层。

注意，在返回输出之前，模型做了一些不寻常的事情：
它运行了一个while循环，在 $L_1$ 范数大于 $1$ 的条件下，
将输出向量除以$2$，直到它满足条件为止。
最后，模型返回了`X`中所有项的和。
注意，此操作可能不会常用于在任何实际任务中，
我们只展示如何将任意代码集成到神经网络计算的流程中。


## 小结

* 一个块可以由许多层组成；一个块可以由许多块组成。
* 块可以包含代码。
* 块负责大量的内部处理，包括参数初始化和反向传播。
* 层和块的顺序连接由`Sequential`块处理。


