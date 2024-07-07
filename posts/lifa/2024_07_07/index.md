# 第八周


## 学习时间
7月1日 到 7月7日

## 学习目标
- 3 paper reviews
- attention mechanisms
    
## 学习内容
- [Research-Progress-on-Binocular-Stereo-Vision-Applications](../../learning/paper_review/Research-Progress-on-Binocular-Stereo-Vision-Applications/)
- [paper_review/survey-on-depth-estimation](../../learning/paper_review/survey-on-depth-estimation/)
- [raft-stereo](../../learning/paper_review/raft-stereo/)


## 学习总结
本周阅读了一篇中文综述一篇英文综述。对双目视觉，基于深度学习的立体匹配涉及到的算法和技术有了一定的了解。
基于深度学习的立体匹配，一般可以分为以下两个大的步骤：
1. 特征提取
2. 正则化以及视差估计

深度估计端到端的方法：

1. 特征学习
2. cost volume 构建
3. 视差计算

细化来说，举一种具体的例子：
1. 从左右两张图中取若干对应的patch
2. 在某个给定视差下，计算左图所有像素点与对应右图之间的相似性
3. 所有视差下的相似性拼接起来作为作为cost volume
4. 在相似性维度，最相似的视差d即为当前像素点的视差
## 下周计划
- 3 paper reviews
- attention mechanisms
