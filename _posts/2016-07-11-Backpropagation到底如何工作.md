---
layout: post
title: "BP到底是如何实现的"
data: 2016-07-11 16:08:48
categories: Deeplearning
---
## How the backpropagation works?
1.基本的符号表达:

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


### 第一个基本方程: 输出层的误差, $\delta^{L}$.

![image](https://github.com/ColdCodeCool/ColdCodeCool.github.io/raw/master/images/bp1.png)

这是一个非常自然的表达式. 等式右边第一项, $\partial C/\partial a_{j}^{L}$, 衡量了cost作为第j个神经元输出激励的函数变化速度. 举例来说, 如果C不是特别依赖神经元j, 那么$\delta_{j}^{L}$将会很小, 这正是我们期望的. 右边第二项, $\sigma^{\prime}(z_{j}^{L})$, 衡量了激励函数$\sigma$在$z_{j}^L$的变化率.

上述方程是$\delta^{L}$一个`componentwise`的表达式, 将其矩阵化或向量化:

![image](https://github.com/ColdCodeCool/ColdCodeCool.github.io/raw/master/images/matrix.png)

这里$\nabla_{a}C$是一个向量, 由$\partial C/\partial a_{j}^L$组成. 我们很容易可以得到$\nabla_{a}C=(a^L - y)$,那么上式可以表示为$\delta^L=(a^L-y)\odot \sigma^{\prime}(z^L)$. 这样一来, 所有符号都是漂亮的向量形式, 那么我们使用Numpy可以很方便地进行计算.

### 第二个基本方程:第l层的误差(与l+1层的误差相关)
第l层误差:

![image](https://github.com/ColdCodeCool/ColdCodeCool.github.io/raw/master/images/lerror.png)

$(w^{l+1})^T$是第l+1层的权值矩阵的转置, 上式看起来很复杂, 但是每个符号都很容易解释. 假设我们知道第l+1层的误差$\delta^{l+1}$, 当我们对l+1层的权值矩阵进行转置时, 我们可以从直觉上认为误差进行了转向, 向反方向传播, 从而给我们提供了对第l层的误差测量. 通过与$\sigma^{\prime}(z^l)$进行hadamard product, 则上述误差就通过激发函数传播给了第l层.

### 第三个基本方程:由bias引起的cost变化率
变化率:

![image](https://github.com/ColdCodeCool/ColdCodeCool.github.io/raw/master/images/biaschange.png)

误差$\delta_{j}^{l}$正好等于第j个neuron的bias, 这对我们来说是个好消息, 因为在前两个方程中, 我们已经计算过第j个neuron的误差.重写BP3, 可表示为$\frac{\partial C}{\partial b}=\delta$.

### 第四个基本方程:由weight引起的cost变化率
有:

![image](https://github.com/ColdCodeCool/ColdCodeCool.github.io/raw/master/images/weightchange.png)

式中的$\delta^{l}$和$a^{l-1}$我们已经知道如何计算. 那么上式可以重写为:$\frac{\partial C}{\partial w}=a_{in}\delta_{out}$, it's understood that $a_{in}$ is the activation of the neuron input to the weight w, and $\delta_{out}$ is the error of the neuron output from the weight w.

由BP1-BP4我们可以得到一些启发, 首先我们从输出层开始. 考虑BP1中的$\delta^{\prime}(z_{j}^{L})$, 回顾sigmoid函数图形, 函数在趋近0和1的时候变得非常平坦, 这以为着前述变化率将非常小, 趋于0. 因此, 我们可以得出结论, 当输出层的激发值趋于0或1时, 最后一层的权值学习将会变得非常慢. 这种情况下的输出层, 我们称其是`saturated`, 对前面网络层也有类似结论.

总结:

![image](https://github.com/ColdCodeCool/ColdCodeCool.github.io/raw/master/images/conclusion.png).

## BP算法
基本流程:

![image](https://github.com/ColdCodeCool/ColdCodeCool.github.io/raw/master/images/process.png)

结合了SGD与mini-batch的BP:

![image](https://github.com/ColdCodeCool/ColdCodeCool.github.io/raw/master/images/bpsgd.png)

具体代码实现在上一篇已经给出.