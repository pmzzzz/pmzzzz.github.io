# pytorch反向传播验证

## 背景
pytorch的反向传播十分方便，对loss 调用backward()就可以，为了更加深刻理解，尝试用一个小例子搞清楚计算的流程，发现跟手动计的结果符合，通过这个例子对pytorch和反向传播都有所掌握

## 要点
1. 需要更新的值加上`requires_grad=True`参数就可以，在之后的计算中，pytorch会自动计算梯度
2. 理论来说，使用`x = x-lr*x.grad`更新值之后，loss就会减小

## 题目
$$
\boldsymbol{x} = [0,1,2,3] \\\
\boldsymbol{y} = \boldsymbol{x}^\top\boldsymbol{x} \\\
y_t = 10 \\\
loss = (y_t-y)^2
$$
## 分析
1. 我们的目标是让$y$与$y_t$更接近，所以用loss来作为接近程度的指标，loss越小，越接近
2. 我们要通过更新$\boldsymbol{x}$的值来使$loss$变小
## 手推

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
## 代码
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
然后呢？更新一次之后再次正向传播计算梯度再次更新权重吗？
