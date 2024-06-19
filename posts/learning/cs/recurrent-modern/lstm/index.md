# 长短期记忆网络LSTM


> 长短记忆网络引入了记忆元，然后用输入门、输出门、遗忘门来控制记忆单元和隐藏状态的输出

## 输入门、忘记门和输出门
{{<image src="io.png">}}  

{{< raw >}}
$$
\begin{aligned}
\mathbf{I}_t &= \sigma(\mathbf{X}_t \mathbf{W}_{xi} + \mathbf{H}_{t-1} \mathbf{W}_{hi} + \mathbf{b}_i),\\
\mathbf{F}_t &= \sigma(\mathbf{X}_t \mathbf{W}_{xf} + \mathbf{H}_{t-1} \mathbf{W}_{hf} + \mathbf{b}_f),\\
\mathbf{O}_t &= \sigma(\mathbf{X}_t \mathbf{W}_{xo} + \mathbf{H}_{t-1} \mathbf{W}_{ho} + \mathbf{b}_o),
\end{aligned}
$$

{{< /raw >}}
$\sigma$是sigmoid 函数，将门的值映射到0到1之间

## 候选记忆元
{{<image src="tildec.png">}}  

{{< raw >}}
$$\tilde{\mathbf{C}}_t = \text{tanh}(\mathbf{X}_t \mathbf{W}_{xc} + \mathbf{H}_{t-1} \mathbf{W}_{hc} + \mathbf{b}_c),$$
{{< /raw >}}
使用tanh函数将$\tilde{C}$映射到-1到1之间

## 记忆元
{{<image src="ct.png">}}  

{{< raw >}}
$$\mathbf{C}_t = \mathbf{F}_t \odot \mathbf{C}_{t-1} + \mathbf{I}_t \odot \tilde{\mathbf{C}}_t.$$

{{< /raw >}}

由遗忘门和输入门分别控制前一时刻和候选记忆元的权重

## 隐状态
{{<image src="h.png">}}

{{< raw >}}
$$\mathbf{H}_t = \mathbf{O}_t \odot \tanh(\mathbf{C}_t).$$

{{< /raw >}}

注意对记忆元进行了tanh确保值域在-1到1之间，然后利用输出门控制当前记忆元多大程度决定隐状态以进行输出以及传递到下一时刻 。

## 代码实现

```python
def lstm(inputs, state, params):
    [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c,
     W_hq, b_q] = params
    (H, C) = state
    outputs = []
    for X in inputs:
        I = torch.sigmoid((X @ W_xi) + (H @ W_hi) + b_i)
        F = torch.sigmoid((X @ W_xf) + (H @ W_hf) + b_f)
        O = torch.sigmoid((X @ W_xo) + (H @ W_ho) + b_o)
        C_tilda = torch.tanh((X @ W_xc) + (H @ W_hc) + b_c)
        C = F * C + I * C_tilda
        H = O * torch.tanh(C)
        Y = (H @ W_hq) + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H, C)
```


