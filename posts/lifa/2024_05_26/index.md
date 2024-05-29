# 第二周

**《Dive-into-DL-PyTorch》** 学习记录(P42-P74)
<!--more-->
## 学习时间
5月20日 到 5月26日
## 学习目标
- [x] 深入了解计算图与自动微分
- 学习第三章的内容，重点学习代码。
    - [x] 线性回归
    - [x] softmax
- 学习第四章MLP
    - [ ] MLP
## 学习内容

### 计算图与自动微分

- [计算图](../../learning/cs/dag/)

### 线性回归
- [线性回归(1)--算法](../../learning/cs/linear_regression1/)
- [线性回归(2)--从零实现](../../learning/cs/linear_regression2/)
- [线性回归(3)--简洁实现](../../learning/cs/linear_regression3/)

### softmax回归
- [softmax(1)--算法](../../learning/cs/softmax1/)
- [softmax(2)--从零实现](../../learning/cs/softmax2/)
- [softmax(3)--简洁实现](../../learning/cs/softmax3/)


## 讨论

### 计算图
作为框架使用者的话，计算图似乎不需要过于深度了解，只需要知道在框架底层，在记录着模型的计算图，以便于更快的(利用GPU并行运算)进行反向传播。

### 线性回归

1. 了解线性回归主要是了解机器学习的大致步骤，其中最重要的是使用了随机梯度下降的方法来更新参数，随机梯度下降法是通用的优化方法，而解析解在很多模型几乎是做不到的
2. 为了更好利用计算机资源，并行计算，不是每次只用一个样本来计算loss，而是每次用一批(mini_batch)
3. 单样本线性回归就是向量乘向量输出标量，多样本就是矩阵乘向量，输出是向量

### softmax
1. 多样本softmax在线性回归基础上，输出是矩阵，所以是矩阵乘矩阵
2. 在 pytorch中softmax和crossentropy集成在了一起,`nn.CrossEntropyLoss()`,直接接受affine层的输出
### 深度学习基本步骤
1. 数据获取(批量迭代) 
```python
import torch.utils.data as Data
batch_size = 10
# 将训练数据的特征和标签组合
dataset = Data.TensorDataset(features, labels)
# 随机读取小批量
data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)
```
2. 模型建立
```python
net = nn.Sequential(
    nn.Linear(num_inputs, 1)
    # 此处还可以传入其他层
    )
```
2. 参数初始化
```python
from torch.nn import init
init.normal_(net[0].weight, mean=0, std=0.01)
```
3. 损失函数
```python
loss = nn.MSELoss()
loss = nn.CrossEntropy()
...
```
4. 优化方法
```python
import torch.optim as optim
optimizer = optim.SGD(net.parameters(), lr=0.03)
```
5. 训练
```python
# loop
# 获取数据
# 前向传播
# loss
# 梯度清零optimizer.zero_grad()
# 计算梯度backword()
# 更新参数optimizer.step() 
# /loop 
```

## 总结

本周学习计划没有全部完成，主要是花了很多时间在博客样式和功能的修改 [博客日志](../../log/)  
也有两天因为懒惰没有学习，总是想着明天学个满的实际很难做到，所以学习还是得细水长流呀，下周要求自己必须每天完成任务，可以少但是不能停。



