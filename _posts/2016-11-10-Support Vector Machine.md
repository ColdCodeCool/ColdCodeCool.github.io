---
layout: post
title: "Support Vector Machine and Sequential Minimal Optimization"
data: 2016-11-10 19:05:43
categories: MachineLearning
---
# 简单笔记
首先考虑用于线性分类器的SVM，对于简单的二分类问题，数据用一个$n$维向量$x$表示，数据对应类别标签记为$y$，标记为-1或1（当然也可以选择其他数字来标记，这个选择只是为后面推导公式方便）。一个线性分类器就是要在$n$维的数据空间中找到一个超平面，其方程可表示为:

$$
w^{T}x + b =0
$$

在二维空间中， 一个超平面就是一条直线。令$f(x)=w^{T}x+b$，显然，如果$f(x)=0$，那么$x$是位于超平面上的点。我们不妨要求所有$f(x)<0$的点，对应的$y$为-1；反之，若$f(x)>0$，记$y$为1。

![image](https://github.com/ColdCodeCool/ColdCodeCool.github.io/raw/master/images/svmplot.png)

如图所示，两种颜色的点分别代表两个类别，红色线表示一个可行的超平面。直观上来看，越接近超平面的点越难分类，因此我们希望两种类别的点距离超平面越远越好。换言之，我们希望找到恰好能将两种类别的点分开，而且两种类别的点到其距离最大的超平面。为此，我们定义functional margin为$\tilde{\gamma}=y(w^{T}x+b)=yf(x)$，注意：前面乘上类别$y$之后可以保证这个margin的非负性(因为若$f(x)<0$，其对应$y$为-1)，而点到超平面的距离定义为geometrical margin。如图所示，

![image](https://github.com/ColdCodeCool/ColdCodeCool.github.io/raw/master/images/svmdis.png)

对于一个点$x$，令其垂直投影到超平面上的对应点为$x_0$，我们有

$$ x=x_0+\gamma \frac{w}{|w|} $$
