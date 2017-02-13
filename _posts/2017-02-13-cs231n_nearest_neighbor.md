---
layout: post
title: "cs231n笔记1"
date: 2017-02-13 11:26:25
categories: cs231n
---
## 重要的tricks--validation.
将训练集切割成training set和validation set，使用training set来训练，然后使用validation set来调参数。因此，validation set is essentially used as a fake test set to tune the hyper-parameters.

## cross-validation
当训练集数据很少，与之相应validation集也很小时，采用一种更为精细的参数调整技术cross-validation。具体方法是将training set切割成特定等分，留一等分做evaluation，其余均作为training set。比如分为5等分调参数k，对每一个k值，对每一等分进行迭代，最终得到5个accuracies，取其平均。最后取accuracy最大的对应k值即可。

但在实际应用中，cross-validation计算量太大，通常只分割出一个validation set。

## nearest neighbor的优缺点
优点：理解简单易实现，不需要训练。缺点：所有计算代价都在test时付出，因为预测原理是与每一个训练样本进行比较。这是一个重大缺点，因为在实际工作中我们往往更关注预测效率，与此相对的另一个极端是神经网络，在训练时花费大量时间，而在测试时很快。