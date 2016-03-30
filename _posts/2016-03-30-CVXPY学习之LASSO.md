---
layout: post
title:  "CVXPY学习之LASSO"
date: 2016-03-30 02:13:52
categories: cvxpy
---
# CVXPY介绍
CVXPY是一款基于python的凸优化问题建模与求解工具。

{% highlight python %}

from cvxpy import *
import numpy
# Problem data.
m = 3
n = 20
numpy.random.seed(1)
A = numpy.random.randn(m, n)
b = numpy.random.randn(m)

# Construct the problem.
x = Variable(n)
objective = Minimize(sum_squares(A*x - b))
constraints = [0 <= x, x <= 1]
prob = Problem(objective, constraints)

# The optimal objective is returned by prob.solve().
result = prob.solve()
# The optimal value for x is stored in x.value.
print x.value
# The optimal Lagrange multiplier for a constraint
# is stored in constraint.dual_value.
print constraints[0].dual_value

{% endhighlight %}
这是一个基本的例子，第一部分设置问题规模，设置seed来确保每次仿真得到相同的随机输入。第二部分为CVXPY的核心描述语言，第三部分是CVXPY内置的问题求解函数，以及结果存放规则。

## LASSO
介绍LASSO之前，先介绍几个基本概念。

### 线性回归
线性回归的模型是这样的，对于一个样本$x_i$，它的输出值是特征的线性组合:
\begin{align}
f(x_i)=\sum_{n=1}^{p}w_{n}x_{in} + w_0=w^{T}x_i
\end{align}
其中w_0成为bias，上式通过增加$x_{i0}=1$把$w_0$也吸收到向量表达中了，上式$x_i$有p+1维。

线性回归的目标是用预测结果尽可能地拟合目标label，用最常见的Least Square作为loss function：
\begin{align}
J(w)=\frac{1}{n}\sum_{i=1}^n(y_i-f(x_i))^2=\frac{1}{n}\lVert y - Xw \rVert^2
\end{align}
可以直接求出最优解:
\begin{align}
w^{*}=(X^{T}X)^{-1}X^{T}y
\end{align}
直接法求解的问题在于上述协方差矩阵$X^{T}X$不可逆时，目标函数最小化必要条件求解方程时有无穷解，无法求出最优解。特别地，当$p>n$时，必然导致上述情况，并且此时也将存在overfitting的问题。此时需要对$w$做一些限制，使得它的最优解空间变小，也就是所谓的regularization。

## Ridge regression
最常见的限制是对$w$的模做约束，如ridge regression,岭回归，也即在线性回归的基础上加上$l_{2}-norm$的约束，loss function变成:
\begin{align}
J_{R}(w)=\frac{1}{2}\lVert y - Xw \rVert^2 + \frac{\lambda}{2}\lVert w \rVert^2
\end{align}
此时有解析解:
\begin{align}
\hat{W_R}=(X^{T}X+\lambda I)^{-1}X^{T}y
\end{align}
其中$\lambda>0$是一个参数，加入正则项之后解就有了一些很好的性质，首先是对$w$的模做约束，使得它的数值会比较小，很大程度上减轻了overfitting的问题，其次是上面的求逆部分肯定可逆，在实际中使用ridge regression通过调节$\lambda$可以得到不同的回归模型。

实际上ridge regression可以用下面的优化目标形式表达：
\begin{align}
& \min_{w} \quad \frac{1}{2} \lVert y - Xw \rVert^2 \\
\begin{align}
& s.t. \lVert w \rVert_{2} < \theta \\
\end{align}
\end{align}
也就是说，我们依然可以优化线性回归的目标，但是条件是$w$的模长不能超过限制$\theta$.