---
layout: post
title: "无线信道CSI学习"
data: 2016-07-13 09:58:04
categories: posts
---
## 信道状态信息(CSI)的概念和作用
信道状态信息在无线通信系统中扮演非常关键的角色, 信号处理中的信道的预编码以及网络操作中的用户小区关联选择都需要知道CSI信息. 目前最流行的(prevalent)的获取CSI的方法是基于导频(pilot-aided)的信道训练, 即在收发端之间发送一系列已知的信号序列, 以此来估计信道响应. 然而, 发送训练序列的代价是信道资源和能量被导频序列消耗了. 

## 基于RNN的信道估计技术用于STBC编码的MIMO系统在瑞利衰落的信道
使用RNN在接收符号序列建立决策域, 信道估计可以当做一个分类任务.

The mobile radio channel can be modeled as a linear time varying channel, where the channel changes with time and distance.