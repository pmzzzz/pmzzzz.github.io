# grade

## Reference
> 我们可以连结一个多元函数对其所有变量的偏导数，以得到该函数的*梯度*（gradient）向量。
> 具体而言，设函数$f:\mathbb{R}^n\rightarrow\mathbb{R}$的输入是
> 一个$n$维向量$\mathbf{x}=[x_1,x_2,\ldots,x_n]^\top$，并且输出是一个标量。
> 函数$f(\mathbf{x})$相对于$\mathbf{x}$的梯度是一个包含$n$个偏导数的向量:
> 
> $$\nabla_{\mathbf{x}} f(\mathbf{x}) = \bigg[\frac{\partial f(\mathbf{x})}{\partial x_1}, \frac{\partial f(\mathbf{x})}{\partial x_2}, \ldots, \frac{\partial f(\mathbf{x})}{\partial x_n}\bigg]^\top,$$
> 
> 其中$\nabla_{\mathbf{x}} f(\mathbf{x})$通常在没有歧义时被$\nabla f(\mathbf{x})$取代。
> 
> 假设$\mathbf{x}$为$n$维向量，在微分多元函数时经常使用以下规则:
> 
> * 对于所有$\mathbf{A} \in \mathbb{R}^{m \times n}$，都有$\nabla_{\mathbf{x}} \mathbf{A} \mathbf{x} = \mathbf{A}^\top$
> * 对于所有$\mathbf{A} \in \mathbb{R}^{n \times m}$，都有$\nabla_{\mathbf{x}} \mathbf{x}^\top \mathbf{A}  = \mathbf{A}$
> * 对于所有$\mathbf{A} \in \mathbb{R}^{n \times n}$，都有$\nabla_{\mathbf{x}} \mathbf{x}^\top \mathbf{A} \mathbf{x}  = (\mathbf{A} + \mathbf{A}^\top)\mathbf{x}$
> * $\nabla_{\mathbf{x}} \|\mathbf{x} \|^2 = \nabla_{\mathbf{x}} \mathbf{x}^\top \mathbf{x} = 2\mathbf{x}$
> 
> 同样，对于任何矩阵$\mathbf{X}$，都有$\nabla_{\mathbf{X}} \|\mathbf{X} \|_F^2 = 2\mathbf{X}$。
> 正如我们之后将看到的，梯度对于设计深度学习中的优化算法有很大用处。

## Understanding
**consider as a function every element**  
example ：

$$
A = \begin{bmatrix}
   1 & 2 \\\
   3 & 4 \\\
   5 & 6
\end{bmatrix}
\boldsymbol{x} = [x_1,x_2]
$$
so：
$$
A\boldsymbol{x} = \begin{bmatrix}
   x_1+2x_2  \\\
   3x_1+4x_2  \\\
   5x_1+6x_2 
\end{bmatrix}
$$
consider as：
$$
f_1(\boldsymbol{x}) = x_1+2x_2\\\
f_2(\boldsymbol{x}) = 3x_1+4x_2\\\
f_3(\boldsymbol{x}) = 5x_1+6x_2\\\
$$

in that：
$$
\nabla f_1 = (1,2)^\top
\nabla f_2 = (3,4)^\top
\nabla f_3 = (5,6)^\top
$$
so：
$$
\nabla f = [\nabla f_1,\nabla f_2,\nabla f_3] = A^\top = \begin{bmatrix}
   1 & 3 & 5  \\\
   2 & 4 & 6
\end{bmatrix}
$$

> [backward方法中gradient参数的意义](https://www.cnblogs.com/Davy-Chen/articles/17317031.html)
