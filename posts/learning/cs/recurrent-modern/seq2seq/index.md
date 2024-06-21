# seq2seq


## 架构
{{<image src="struc.png">}}

- 编码器是一个RNN，读取输入句子 ，可以是双向RNN
- 解码器使用另一个RNN来输出

## 训练过程
{{<image src="train.png">}}
- 将输入按照顺序输入给编码器
- 将编码器的最后一个时刻、最后一层的隐藏状态传给解码器，作为解码器的初始状态
- 将目标句子的单词一个个输入给解码器，输出层为实际的下一个词语
- 反向传播

## 预测过程
{{<image src="predic.png">}}
- 将输入按照顺序输入给编码器
- 将编码器的最后一个时刻、最后一层的隐藏状态传给解码器，作为解码器的初始状态
- 将`<bos>`传给解码器，进行循环预测
- 直到预测到`<eos>`预测完成

## 模型评估

我们将BLEU定义为：

{{< raw >}}
$$
\exp\left(\min\left(0, 1 - \frac{\mathrm{len}_{\text{label}}}{\mathrm{len}_{\text{pred}}}\right)\right) \prod_{n=1}^k p_n^{1/2^n},
$$
其中$\mathrm{len}_{\text{label}}$表示标签序列中的词元数和
$\mathrm{len}_{\text{pred}}$表示预测序列中的词元数，
$k$是用于匹配的最长的$n$元语法。
另外，用$p_n$表示$n$元语法的精确度，它是两个数量的比值：
第一个是预测序列与标签序列中匹配的$n$元语法的数量，
第二个是预测序列中$n$元语法的数量的比率。
具体地说，给定标签序列$A$、$B$、$C$、$D$、$E$、$F$
和预测序列$A$、$B$、$B$、$C$、$D$，
我们有$p_1 = 4/5$、$p_2 = 3/4$、$p_3 = 1/3$和$p_4 = 0$。

{{< /raw >}}
```python
def bleu(pred_seq, label_seq, k):  #@save
    """计算BLEU"""
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[' '.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[' '.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[' '.join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score
```
