---
layout: post
title:  "CVXPY"
date: 2016-03-30 14:06:02
categories: cvxpy
---
## CVXPY介绍
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


x = Variable(n)
objective = Minimize(sum_squares(A*x - b))
constraints = [0 <= x, x <= 1]
prob = Problem(objective, constraints)


result = prob.solve()

print x.value

print constraints[0].dual_value

{% endhighlight %}