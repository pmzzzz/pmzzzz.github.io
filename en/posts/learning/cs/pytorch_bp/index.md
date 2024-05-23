# BP Validation in Pytorch

## Backgroud
It's very easy to do BP in Pytorch , just call backward() with loss .  
to further my understanding , I'll try to figure out the compute procces with a little example.  
Turns out that doing it with my hand has the same result with pytorch.  
I`ve got some basic understanding about BP and Pytorch 

## Key Point
1. add parameter`requires_grad=True` to Tensor who need to be updated,pytorch will caculate gradiant automaticly.
2. Theoretically,update x with `x = x-lr*x.grad`,loss will be closer to 0.

## Quez
$$
\boldsymbol{x} = [0,1,2,3] \\\
\boldsymbol{y} = \boldsymbol{x}^\top\boldsymbol{x} \\\
y_t = 10 \\\
loss = (y_t-y)^2
$$
## Analyze
1. our goal is let $y$ closer to $y_t$,so we use loss to indicate that ，less loss，better result
2. we shoud lower $loss$ by update $\boldsymbol{x}$
## Hand Caculate

$$
y = \boldsymbol{x}^\top\boldsymbol{x} = x_1^2 + x_2^2 + x_3^2 + x_4^2=14\\\
\frac{\partial loss}{\partial y} = 2(y-y_t)=8\\\
\frac{\partial loss}{\partial x_1} = \frac{\partial loss}{\partial y}·\frac{\partial y}{\partial x_1} = 4(y-y_t)x_1 = 0\\\
\frac{\partial loss}{\partial x_2} = \frac{\partial loss}{\partial y}·\frac{\partial y}{\partial x_2} = 4(y-y_t)x_2 = 16\\\
\frac{\partial loss}{\partial x_3} = \frac{\partial loss}{\partial y}·\frac{\partial y}{\partial x_3} = 4(y-y_t)x_3= 32\\\
\frac{\partial loss}{\partial x_4} = \frac{\partial loss}{\partial y}·\frac{\partial y}{\partial x_4} = 4(y-y_t)x_4 = 48 
$$
by that we got all the gradiants of x on loss,update x by a given learn-rate
$$
x_n = x_n-lr*\frac{\partial loss}{\partial x_n} 
$$
in this case ，`lr = 0.01`,so after updating:
$$
\boldsymbol{x} = [0,0.84,1.68, 2.52]
$$
caculate $y$ and $loss$ again:
$$
y = \boldsymbol{x}^\top\boldsymbol{x} = x_1^2 + x_2^2 + x_3^2 + x_4^2 = 9.8784 \\\
loss = (y_t-y)^2 = 0.0148
$$
as expatation，$y$ is closer to 10，and less $loss$
## Code
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
## Mystery
What's next? how to BP again?

