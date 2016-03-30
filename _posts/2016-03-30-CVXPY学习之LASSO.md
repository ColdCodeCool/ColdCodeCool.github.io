---
layout: post
title:  "CVXPY"
date: 2016-03-30 02:13:52
categories: cvxpy
---
# CVXPY介绍
CVXPY是一款基于python的凸优化问题建模与求解工具。

{% highlight python %}

from cvxpy import *
import numpy
# Problem data.
m = 30
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
