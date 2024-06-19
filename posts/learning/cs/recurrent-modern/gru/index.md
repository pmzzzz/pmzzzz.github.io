# 门控单元GRU


> 门控单元的核心思想是控制隐状态的生成和传递

## 重置门和更新门

{{<image src="gate1.png">}}
{{< raw >}}
$$
\begin{aligned}
\mathbf{R}_t = \sigma(\mathbf{X}_t \mathbf{W}_{xr} + \mathbf{H}_{t-1} \mathbf{W}_{hr} + \mathbf{b}_r),\\
\mathbf{Z}_t = \sigma(\mathbf{X}_t \mathbf{W}_{xz} + \mathbf{H}_{t-1} \mathbf{W}_{hz} + \mathbf{b}_z),
\end{aligned}
$$
{{< /raw >}}

可以看出，$R_t$ 和 $Z_t$ 是完全相同的，它们有各自的参数，决定它们不同的是后续的操作。

## 候选隐状态

{{<image src="h.png">}}

{{< raw >}}
$$\tilde{\mathbf{H}}_t = \tanh(\mathbf{X}_t \mathbf{W}_{xh} + \left(\mathbf{R}_t \odot \mathbf{H}_{t-1}\right) \mathbf{W}_{hh} + \mathbf{b}_h),$$
{{< /raw >}}

可以看出，候选隐状态受 $R_t$ 的影响，具体影响方式是决定了上一时间步的隐藏状态多大程度被保留，如果 $R_t=1$ 那么完全保留，$R_t=0$ 则完全舍弃

## 隐状态

{{<image src="gru.png">}}

{{< raw >}}
$$\mathbf{H}_t = \mathbf{Z}_t \odot \mathbf{H}_{t-1}  + (1 - \mathbf{Z}_t) \odot \tilde{\mathbf{H}}_t.$$
{{< /raw >}}
可以看出，$Z_t$ 决定了 前一时间步和当前候选隐藏状态的权重，并加和得到当前的隐状态，如果$Z_t=1$ 那么完全完全使用前一个状态，$Z_t=0$则完全更新当前状态为 $\tilde{H}$


## 代码实现
几乎就是将上述公式复刻一下，注意这里的输出Y没有加上softmax层，需要在后续模型中加上。
```python
def gru(inputs, state, params):
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        Z = torch.sigmoid((X @ W_xz) + (H @ W_hz) + b_z)
        R = torch.sigmoid((X @ W_xr) + (H @ W_hr) + b_r)
        H_tilda = torch.tanh((X @ W_xh) + ((R * H) @ W_hh) + b_h)
        H = Z * H + (1 - Z) * H_tilda
        Y = H @ W_hq + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,)
```


