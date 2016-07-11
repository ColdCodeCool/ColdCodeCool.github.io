---
layout: post
title: "BP到底是如何实现的"
data: 2016-07-11 16:08:48
categories: Deeplearning
---
## How the backpropagation works?
基本的符号表达:

![image](https://github.com/ColdCodeCool/ColdCodeCool.github.io/raw/master/images/notation.png)

令$a_{j}^{l}$表示$l^{th}$层第$j^{th}$个神经元的激发值, 则有:

![image](https://github.com/ColdCodeCool/ColdCodeCool.github.io/raw/master/images/activation.png)

求和部分是对第$(l-1)^{th}$层所有神经元. 将其向量化, 得到:

![image](https://github.com/ColdCodeCool/ColdCodeCool.github.io/raw/master/images/vectorization.png)

得到某层激励如何与上一层激励产生联系的关系式. 我们将中间量$z^l = w^{l}a^{l-1} + b^l$带入上式, 其中$z^l$称为第l层神经元的加权输入, 得到$a^l = \sigma(z^l)$.

## 两个cost function的假设
二次的cost function:

![image](https://github.com/ColdCodeCool/ColdCodeCool.github.io/raw/master/images/costfunction.png)

第一个假设:cost function要能够写成所有单个训练样本cost的平均:$C=\frac{1}{n}C_x$. 第二个假设, cost能写成$C = C(a^L)$.

## Hadamard product
假设$s$和$t$是两个相同维度的向量, 那么$s\odot t$表示`elementwise`的乘积. 举例来说:

![image](https://github.com/ColdCodeCool/ColdCodeCool.github.io/raw/master/images/hadamard.png)

成为Hadamard product 或 Schur product.

## 四个backpropagation背后的基本方程
BP是帮助我们理解怎样通过改变weights和biases来改变cost function的方法, 归根到底, 这意味着计算偏导$\partial C/\partial w_{jk}^l$和$\partial C/\partial b_{j}^l$. 但在这之前, 我们先引入中间量$\delta_{j}^l$, 我们称其为第l层第j个神经元的误差:

![image](https://github.com/ColdCodeCool/ColdCodeCool.github.io/raw/master/images/error.png)

第一个基本方程: 输出层的误差, $\delta^{L}$.

![image](https://github.com/ColdCodeCool/ColdCodeCool.github.io/raw/master/images/bp1.png)

这是一个非常自然的表达式. 等式右边第一项, $\partial C/\partial a_{j}^{L}$, 衡量了cost作为第j个神经元输出激励的函数变化速度. 举例来说, 如果C不是特别依赖神经元j, 那么$\delta_{j}^{L}$将会很小, 这正是我们期望的. 右边第二项, $\sigma^{\prime}(z_{j}^{L})$, 衡量了激励函数$\sigma$在$z_{j}^L$的变化率.

上述方程是$\delta^{L}$一个`componentwise`的表达式, 将其矩阵化或向量化:

![image](https://github.com/ColdCodeCool/ColdCodeCool.github.io/raw/master/images/matrix.png)

这里$\nabla_{a}C$是一个向量, 由$\partial C/\partial a_{j}^L$组成. 我们很容易可以得到$\nabla_{a}C=(a^L - y)$,那么上式可以表示为$\delta^L=(a^L-y)\odot \sigma^{\prime}(z^L)$.