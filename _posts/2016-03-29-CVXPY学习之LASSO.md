---
layout: post
title:  "CVXPY学习之LASSO"
date: 2016-03-30 00:00:13
categories: CVXPY学习
---
#CVXPY介绍
CVXPY是一个python版本的凸优化问题建模及求解工具，作者是凸优化大牛Stephen Boyd 的学生Steven Diamond。它以一种更加接近自然语言的方式对问题进行描述建模，相比其他求解工具，没有那么严格的标准化要求。一个简单的凸优化问题建模与求解例子如下：

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
设置seed是为了每次仿真得到同样的随机数，保证仿真的有效性。中间部分以CVXPY的风格描述了一个最小二乘问题，x为要求解的变量，为nx1的行向量，优化目标objective以CVXPY的简洁风格描述，constraints为问题约束条件，prob以优化目标和约束条件的形式呈现。最后，result使用CVXPY自带的solver求解，约束条件的最优拉格朗日乘子由constraint.dual_value给出。

## LASSO


