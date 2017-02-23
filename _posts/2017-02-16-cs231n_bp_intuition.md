---
layout: post
title: "cs231n笔记3"
date: 2017-02-16 14:59:17
categories: cs231n
---
## Gradients for vectorized operations

### Matrix-Matrix multiply gradient
$$
W = np.random.randn(5,10)
X = np.random.randn(10,3)
D = W.dot(X)

dD = np.random.randn(*D.shape)
dW = dD.dot(X.T)
dX = W.T.dot(dD)
$$

这种情形下，不需要死记表达式，只需进行dimension analysis即可。

## batch normalization
使得深度神经网络好训练的方法有两种，第一是采用sgd+momentum,adagrad,adam这种优化算法，第二是改变网络结构。batch normalization是最近提出来的一种有效方法，思想很简单，机器学习算法在输入数据包含不相关的特征并服从标准正态分布(均值为0，方差为1)时效果往往会更好。因此，在预处理步骤我们可以事先decorrelate输入数据的features，以使第一层网络接收到很好分布的数据，但更深层的网络将不再接收很好地标准正态分布，这时我们引入batch normalization。即在训练时，用一个mini-batch的数据来估计均值和方差，然后将其标准正态化。这种处理方式也有可能弱化神经网络的表征能力，因为可能会有一些网络层接收非标准正态分布的数据更好。

## dropout
dropout is a technique for regularizing neural networks by randomly setting some features to zero during the forward pass.
