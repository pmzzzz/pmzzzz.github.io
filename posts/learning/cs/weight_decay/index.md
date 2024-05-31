# 权重衰减与正则化


在训练参数化机器学习模型时，
*权重衰减*（weight decay）是最广泛使用的正则化的技术之一，
它通常也被称为$L_2$*正则化*。

## 正则化

正则化(Regularization)指的是通过引入噪声或限制模型的复杂度，降低模型对输入或者参数的敏感性，避免过拟合，提高模型的泛化能力。常用的正则化方法包括约束目标函数(等价于约束模型参数)、约束网络结构、约束优化过程。

- 约束目标函数：在目标函数中增加模型参数的正则化项，包括L2正则化, L1正则化, 弹性网络正则化, L0正则化, 谱正则化, 自正交性正则化, WEISSI正则化, 梯度惩罚
- 约束网络结构：在网络结构中添加噪声，包括随机深度, Dropout及其系列方法,
- 约束优化过程：在优化过程中施加额外步骤，包括数据增强, 梯度裁剪, Early Stop, 标签平滑, 变分信息瓶颈, 虚拟对抗训练, Flooding

> [正则化](https://0809zheng.github.io/2020/03/03/regularization.html)

## 权重衰减（L2正则化）

由上可知，权重衰减是一种通过约束目标函数，限制模型的复杂度来缓解过拟合问题的方式

原本的损失函数如下：

$$L(\mathbf{w}, b) = \frac{1}{n}\sum_{i=1}^n \frac{1}{2}\left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right)^2.$$

现在在损失函数加上权重的L2范式：

$$L(\mathbf{w}, b) + \frac{\lambda}{2} \||\mathbf{w}\||^2,$$

> 对于$\lambda = 0$，我们恢复了原来的损失函数。
对于$\lambda > 0$，我们限制$\| \mathbf{w} \|$的大小。
这里我们仍然除以$2$：当我们取一个二次函数的导数时，
$2$和$1/2$会抵消，以确保更新表达式看起来既漂亮又简单。
为什么在这里我们使用平方范数而不是标准范数（即欧几里得距离）？
我们这样做是为了便于计算。
通过平方$L_2$范数，我们去掉平方根，留下权重向量每个分量的平方和。
这使得惩罚的导数很容易计算：导数的和等于和的导数。

此时参数更新的方式变为：


$$
\begin{aligned}
\mathbf{w} & \leftarrow \left(1- \eta\lambda \right) \mathbf{w} - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \mathbf{x}^{(i)} \left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right).
\end{aligned}
$$

## 关键代码
L2正则化实际上就是给loss加上了一部分，所以关键代码如下：
```python
def l2_penalty(w):
    return torch.sum(w.pow(2)) / 2
...
l = loss(net(X), y) + lambd * l2_penalty(w)
...
```
其余的训练步骤与之前无异。
