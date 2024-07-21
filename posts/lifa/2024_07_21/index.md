# 第十周


## 学习时间
7月15日 到 7月21日

## 学习目标
- d2l注意力机制
    
## 学习内容
- [bahdanau](../../learning/cs/attention-mechanisms/bahdanau-attention/)
- [多头注意力](../../learning/cs/attention-mechanisms/multihead-attention/)
- [子注意力与位置编码](../../learning/cs/attention-mechanisms/self-attention-and-positional-encoding/)
- [Transformer](../../learning/cs/attention-mechanisms/transformer/)

## 学习总结
本周恢复了学习的脚步，将d2l的注意力内容进行了补充，注意力机制可以简单的理解为加权平均，而所谓的多头注意力就是进行多次加权平均

transformer的编码解码器结构大致相同，解码器多出来的部分主要为了满足自回归的属性

个人认为，无论是RNN的发展，还是注意力机制的发展和优化，参数量都在增加，也许一个模型的表达能力和参数量有很大的正相关关系，一些大模型用参数量来标榜自己的能力，这个现象能解释我的猜测。

也许可以说，改进深度学习的模型，本质上就是在原有模型的基础上增加合适的结构，使之参数量增加，从而表达能力增加，从而提升效果。

这么看来resnet的含金量还在不断提升，它给了模型“无痛增加参数量”的机会。

## 下周计划
- d2l计算机视觉内容
- 双目视觉领域综述
