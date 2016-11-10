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

如图所示，两种颜色的点分别代表两个类别，红色线表示一个可行的超平面。直观上来看，越接近超平面的点越难分类，因此我们希望两种类别的点距离超平面越远越好。换言之，我们希望找到恰好能将两种类别的点分开，而且两种类别的点到其距离最大的超平面。为此，我们定义functional margin为$\hat{\gamma}=y(w^{T}x+b)=yf(x)$，注意：前面乘上类别$y$之后可以保证这个margin的非负性(因为若$f(x)<0$，其对应$y$为-1)，而点到超平面的距离定义为geometrical margin。如图所示，

![image](https://github.com/ColdCodeCool/ColdCodeCool.github.io/raw/master/images/svmdis.png)

对于一个点$x$，令其垂直投影到超平面上的对应点为$x_0$，我们有

$$ x=x_0+\gamma \frac{w}{\|w\|} $$

又$x_0$是超平面上的点，满足$f(x_0)=0$，代入超平面的方程即可算出

$$\gamma=\frac{w^{T}x+b}{|w|}=\frac{f(x)}{\|w\|}$$

不过，这里的$\gamma$是带符号的，我们需要的只是它的绝对值，因此类似于functional margin，我们也使其乘上对应类别$y$。于是geometrical margin定义为

$$\tilde{\gamma}=y\gamma=\frac{\hat{\gamma}}{\|w\|}$$

显然，functional margin和geometrical margin相差一个缩放因子$\Vert w\Vert$。按照我们前面的分析，对于一个数据点进行分类，当它的margin越大的时候，分类的confidence越大。对于一个包含$n$个点的数据集，我们可以很自然地定义它的margin为所有这$n$个点的margin值中最小的那个。于是，为了使得分类的confidence高，我们希望所选择的hyper plane能最大化这个margin值。不过我们有两个margin可以选，但functional margin明显不适合用来最大化的，因为在hyper plane固定以后，我们可以等比例地缩放$w$的长度和$b$的值，这样可以使得$f(x)=w^{T}x+b$的值任意大，亦即functional margin可以在hyper plane保持不变的情况下被取得任意大，而geometrical margin则没有这个问题，因为分子上为$\Vert w\Vert$，缩放$w$和$b$不会改变$\tilde{\gamma}$。它只随着hyper plane的变动而变动，因此，这是更加合适的一个margin。这样一来，我们的maximum margin classifier的目标函数即定义为

$$ \max \tilde{\gamma} $$

还需要满足一些条件

$$y_{i}(w^{T}x_i+b)=\hat{\gamma_i}\geq \hat{\gamma}，i=1，\ldots，n$$

其中$\hat{\gamma}=\tilde{\gamma}\|w\|$，根据之前的讨论，即使在超平面固定的情况下，$\hat{\gamma}$的值也可以随着$\|w\|$的变化而变化。由于我们的目标就是要确定超平面，因此可以把这个无关的变量固定下来，固定的方式有两种：一是固定$\|w\|$，当我们找到最有的$\tilde{\gamma}$时$\hat{\gamma}$也就可以随之而固定；二是反过来固定$\hat{\gamma}$，此时$|w|$也可以根据最优的$\tilde{\gamma}$得到。出于方便推导和优化的目的，我们选择第二种，令$\hat{\gamma}=1$，则我们的目标函数化为：

$$ \max \frac{1}{|w|}， s.t.，y_{i}(w^{T}x+b)\geq 1，i=1，\ldots，n$$

通过求解这个问题，我们就可以找到一个margin最大的classifier。如下图所示，

![image](https://github.com/ColdCodeCool/ColdCodeCool.github.io/raw/master/images/svmplot1.png)

中间的红色线条是Optimal Hyper Plane，另外两条线到红线的距离都是$\tilde{\gamma}$。至此，我们介绍了maximum margin classifier，对于给定数据集，我们找到了能划分最小margin的最优hyper plane，即得到了max min margin, $w.r.t w$ and $b$。可以看到两个支撑着中间gap的超平面，它们到中间的separating hyper plane的距离相等并且等于我们所能得到的最大的geometrical margin。而"支撑"这两个超平面的必然会有一些点，试想，如果某超平面没有碰到任意一个点的话，那么我们就可以进一步地扩充中间的gap，于是这个就不是最大的margin了。由于在$n$维向量空间里一个点实际上是和以原点为起点，该点为终点的一个向量等价，所以这些支撑的点就叫做支持向量。

事实上，当最优的超平面确定下来之后，除了支持向量以外的其他点都不会对超平面产生任何影响，这种特性在实际当中的一个最直接的好处在于存储和计算上的优越性。例如，使用100万个点求出一个超平面，其中是supporting vector的点就只有100个，那么后续计算只需要记住这100个点的信息即可。（通常，除了K-Nearest Neighbor之类的Memory-based learning，其他算法也都不需要记忆所有点的信息）。

上面我们得出了优化问题，其等价形式为

$$ \min \frac{1}{2}|w|^2，s.t.，y_{i}(w^{T}x_i+b)\geq 1，i=1，\ldots，n$$