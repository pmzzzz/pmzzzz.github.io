# 第一周

**《Dive-into-DL-PyTorch》** 学习记录(P0-P41)
<!--more-->
## 学习时间
5月13日 到 5月19日
## 学习目标
1. 深度学习起源、发展、特点
2. 学习环境搭建
3. PyTorch基础认识和使用
## 学习内容
### 神经网络的核心
- 交替使⽤线性处理单元与⾮线性处理单元，它们经常被称为“层”
- 使⽤链式法则（即反向传播）来更新⽹络的参数

### Tensor基础
- 常用创建Tenso：empty,rand,zeros,ones,arange,randn_like,eye,linspace
- `reshape` 改变形状
- 使⽤ `.numpy()` 将 Tensor 转换成NumPy数组 
- 使用`torch.from_numpy()` 将NumPy数组转换成 Tensor 
- 索引出来的结果与原数据共享内存，也即修改⼀个，另⼀个会跟着修改
- `torch.matmul()` 可以表示矩阵/向量各种情况相乘
### 梯度
- 输入输出都为向量的函数 , 那么y关于x的梯度就是⼀个雅可⽐矩阵（Jacobian matrix）
> [雅可比矩阵](https://zh.wikipedia.org/wiki/%E9%9B%85%E5%8F%AF%E6%AF%94%E7%9F%A9%E9%98%B5)
### Tensor的自动求梯度
- 将Tensor属性 `.requires_grad` 设置为 `True`即可自动追踪梯度
- `.backward()` 来完成所有梯度计算。此 Tensor 的梯度将累积到 `.grad` 属性中
- 要求对标量求使用`.backward()`，否则，需要传⼊⼀个与 y 同形的 Tensor

## 讨论
### torch中的向量
在学线性代数的时候，行向量和列向量对我们来说是不同的也就是说$\boldsymbol{a}^\top \boldsymbol{b}$和$\boldsymbol{a} \boldsymbol{b}^\top$的结果是不同的，前者会得到一个秩为1的**矩阵**而后者会得到一个**实数**。  
那么在pytorch中是否也能这么做呢？  
no！pytorch中，向量是*自适应*的：
```python
x = torch.arange(3)
y = torch.arange(3)
torch.dot(x,y) == torch.dot(y,x) # tensor(True)
```
以上代码将x，y调换了位置，但是结果相同，所以，可以说，在前面的向量会被认为是行向量，在后面的会被认为是列向量

### 梯度
> 输入输出都为向量的函数 , 那么y关于x的梯度就是⼀个雅可⽐矩阵（Jacobian matrix）  

**也可以理解为向量的每一个元素就是一个函数的输出**  
例如 ：

$$
A = \begin{bmatrix}
   1 & 2 \\\
   3 & 4 \\\
   5 & 6
\end{bmatrix}
\boldsymbol{x} = [x_1,x_2]
$$
有：
$$
A\boldsymbol{x} = \begin{bmatrix}
   x_1+2x_2  \\\
   3x_1+4x_2  \\\
   5x_1+6x_2 
\end{bmatrix}
$$
可以理解为：
$$
f_1(\boldsymbol{x}) = x_1+2x_2\\\
f_2(\boldsymbol{x}) = 3x_1+4x_2\\\
f_3(\boldsymbol{x}) = 5x_1+6x_2\\\
$$

其中：
$$
\nabla f_1 = (1,2)^\top
\nabla f_2 = (3,4)^\top
\nabla f_3 = (5,6)^\top
$$
所以：
$$
\nabla f = [\nabla f_1,\nabla f_2,\nabla f_3] = A^\top = \begin{bmatrix}
   1 & 3 & 5  \\\
   2 & 4 & 6
\end{bmatrix}
$$



### torch的反向传播
pytorch的反向传播十分方便，对loss 调用backward()就可以，为了更加深刻理解，尝试用一个小例子搞清楚计算的流程，发现跟手动计的结果符合，通过这个例子对pytorch和反向传播都有所掌握
#### 题目
$$
\boldsymbol{x} = [0,1,2,3] \\\
\boldsymbol{y} = \boldsymbol{x}^\top\boldsymbol{x} \\\
y_t = 10 \\\
loss = (y_t-y)^2
$$
#### 分析
1. 我们的目标是让$y$与$y_t$更接近，所以用loss来作为接近程度的指标，loss越小，越接近
2. 我们要通过更新$\boldsymbol{x}$的值来使$loss$变小
#### 手推

$$
y = \boldsymbol{x}^\top\boldsymbol{x} = x_1^2 + x_2^2 + x_3^2 + x_4^2=14\\\
\frac{\partial loss}{\partial y} = 2(y-y_t)=8\\\
\frac{\partial loss}{\partial x_1} = \frac{\partial loss}{\partial y}·\frac{\partial y}{\partial x_1} = 4(y-y_t)x_1 = 0\\\
\frac{\partial loss}{\partial x_2} = \frac{\partial loss}{\partial y}·\frac{\partial y}{\partial x_2} = 4(y-y_t)x_2 = 16\\\
\frac{\partial loss}{\partial x_3} = \frac{\partial loss}{\partial y}·\frac{\partial y}{\partial x_3} = 4(y-y_t)x_3= 32\\\
\frac{\partial loss}{\partial x_4} = \frac{\partial loss}{\partial y}·\frac{\partial y}{\partial x_4} = 4(y-y_t)x_4 = 48 
$$
这样就求出了各个x对于loss梯度，只要按照一定的学习率更新参数就可以
$$
x_n = x_n-lr*\frac{\partial loss}{\partial x_n} 
$$
在这个例子中，lr = 0.01,所以更新之后:
$$
\boldsymbol{x} = [0,0.84,1.68, 2.52]
$$
再次计算$y$和$loss$：
$$
y = \boldsymbol{x}^\top\boldsymbol{x} = x_1^2 + x_2^2 + x_3^2 + x_4^2 = 9.8784 \\\
loss = (y_t-y)^2 = 0.0148
$$
可以发现确实如我们所料，y接近10了，loss变小了
#### 代码
```python
import torch
lr = 0.01
x = torch.arange(4.0,requires_grad=True)
y = torch.dot(x,x) # y= 14
y_target = 10
loss = (y_target-y)**2
# loss = tensor(16., grad_fn=<PowBackward0>)
loss.backward()
# x.grad = tensor([ 0., 16., 32., 48.])
x = x-lr*x.grad
y = torch.dot(x,x) # y = 9.8784
loss = (y_target-y)**2
#loss = tensor(0.0148, grad_fn=<PowBackward0>)
```
## 疑问
在使用`x = x-lr*x.grad`更新x后，再查看x的梯度就会报错，那如何进行进一步的梯度下降呢。
## 下期计划

- 深入了解一下pytorch中的计算图
- 继续学习第三章

