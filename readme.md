# 复现了发表在MM2018上的一篇论文中的算法
## 1.summary
文章提出了一种新的哈希算法，并验证了robustness 和discrimination 上的performance

## 2.工具
用matlab脚本做图像攻击批量生成了大部分相似图片，还有一些用photoshop的批量功能生成。

## 3.编程语言
用python编程的，计算速度慢，因此用了python自带的concurrent库进行多进程编程。一个图像的哈希计算过程被当成一个任务
