---
layout: post
title: "Support Vector Machine and Sequential Minimal Optimization"
data: 2016-11-10 19:05:43
categories: MachineLearning
---
# 简单笔记
需要\color{red}{声明}本文内容基本整理自pluskid的SVM博客内容，链接地址在[这](http://blog.pluskid.org/?p=685)。
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

其中$\hat{\gamma}=\tilde{\gamma}\|w\|$，根据之前的讨论，即使在超平面固定的情况下，$\hat{\gamma}$的值也可以随着$\|w\|$的变化而变化。由于我们的目标就是要确定超平面，因此可以把这个无关的变量固定下来，固定的方式有两种：一是固定$\|w\|$，当我们找到最有的$\tilde{\gamma}$时$\hat{\gamma}$也就可以随之而固定；二是反过来固定$\hat{\gamma}$，此时$\|w\|$也可以根据最优的$\tilde{\gamma}$得到。出于方便推导和优化的目的，我们选择第二种，令$\hat{\gamma}=1$，则我们的目标函数化为：

$$ \max \frac{1}{\|w\|}， s.t.，y_{i}(w^{T}x+b)\geq 1，i=1，\ldots，n$$

通过求解这个问题，我们就可以找到一个margin最大的classifier。如下图所示，

![image](https://github.com/ColdCodeCool/ColdCodeCool.github.io/raw/master/images/svmplot1.png)

中间的红色线条是Optimal Hyper Plane，另外两条线到红线的距离都是$\tilde{\gamma}$。至此，我们介绍了maximum margin classifier，对于给定数据集，我们找到了能划分最小margin的最优hyper plane，即得到了max min margin, $w.r.t w$ and $b$。可以看到两个支撑着中间gap的超平面，它们到中间的separating hyper plane的距离相等并且等于我们所能得到的最大的geometrical margin。而"支撑"这两个超平面的必然会有一些点，试想，如果某超平面没有碰到任意一个点的话，那么我们就可以进一步地扩充中间的gap，于是这个就不是最大的margin了。由于在$n$维向量空间里一个点实际上是和以原点为起点，该点为终点的一个向量等价，所以这些支撑的点就叫做支持向量。

事实上，当最优的超平面确定下来之后，除了支持向量以外的其他点都不会对超平面产生任何影响，这种特性在实际当中的一个最直接的好处在于存储和计算上的优越性。例如，使用100万个点求出一个超平面，其中是supporting vector的点就只有100个，那么后续计算只需要记住这100个点的信息即可。（通常，除了K-Nearest Neighbor之类的Memory-based learning，其他算法也都不需要记忆所有点的信息）。

上面我们得出了优化问题，其等价形式为

$$ \min \frac{1}{2}\|w\|^2，s.t.，y_{i}(w^{T}x_i+b)\geq 1，i=1，\ldots，n$$

很明显这是一个凸优化问题，或者更具体地说，它是一个二次优化问题，目标函数是二次的，约束条件是线性的。这个问题可以用现成的（Quadratic Programming）优化包求解，比如CVXOPT。但是我们通过拉格朗日对偶变换得到对偶问题的一些性质可以使得问题能够更加高效地解决，具体Lagrangian形式为

$$ \mathcal{L}(w,b,\alpha)=\frac{1}{2}\|w\|^2-\sum_{i=1}^{n}\alpha_{i}(y_{i}(w^{T}x_{i}+b)-1)$$

然后我们令

$$\theta(w)=\max_{\alpha_i\geq 0}\mathcal{L}(w,b,\alpha)$$

容易验证，当某个约束条件不满足时，例如$y_{i}(w^{T}x_{i}+b)<1$，那么我们显然有$\theta(w)=\infty$(只要令$\alpha_i=\infty$即可)。而当所有约束条件都满足时，则有$\theta(w)=\frac{1}{2}\|w\|^2$，亦即我们最初要最小化的量。因此，在要求约束条件得到满足的情况下最小化$\frac{1}{2}\|w\|^2$实际上等价于直接最小化$\theta(w)$(当然，这里也有约束条件，就是$\alpha_i\geq 0，i=1，\ldots，n$)，因为如果约束条件没有得到满足，$\theta(w)$会等于无穷大，自然不是我们所要求的最小值。具体写出来，我们现在的目标函数变成了

$$\min_{w,b}\theta(w)=\min_{w,b}\max_{\alpha_i\geq 0}\mathcal{L}(w,b,\alpha)=p^*$$

这里用$p^*$表示这个问题的最优值，这个问题和我们最初的问题是等价的。不过，我们现在来把最大和最小的位置交换一下：

$$\max_{\alpha_i\geq 0}\min_{w,b}\mathcal{L}(w,b,\alpha)=d^*$$

当然，交换以后的问题不再等价于原问题，这个新问题的最优值用$d^*$来表示。并且，我们有$d^{*}\leq p^*$，总之第二个问题的最优值在这里提供了一个第一个问题的最优值的一个下界，在满足某些条件的情况下，这两者相等，这时我们就可以通过求解第二个问题间接求解第一个问题。具体来说，就是要满足KKT条件。

首先要让$\mathcal{L}$关于$w$和$b$最小化，我们分别令$\partial {\mathcal{L}}/\partial{w}$和$\partial {\mathcal{L}}/\partial{b}$等于零：

$$
\begin{align}
& \frac{\partial{\mathcal{L}}}{\partial{w}}=0\Rightarrow w=\sum_{i=1}^{n}\alpha_{i}y_{i}x_i\\
& \frac{\partial{\mathcal{L}}}{\partial{b}}=0\Rightarrow \sum_{i=1}^{n}\alpha_{i}y_i
\end{align}
$$

带回得到关于dual variable $\alpha$的优化问题：

$$
\begin{align}
\max_{\alpha} \quad \sum_{i=1}^{n}\alpha_{i}-\frac{1}{2}\sum_{i,j=1}^{n}\alpha_{i}\alpha_{j}y_{i}y_{j}x_{i}^{T}x_j
\end{align}
$$

$$
\begin{align*}
&\begin{array}{t}
s.t.，& \alpha_{i}\geq 0，i=1,\ldots，n\\
      & \sum_{i=1}^{n}\alpha_{i}y_i
\end{array}
\end{align*}
$$

这个问题有更加高效的解法（SMO），先不做介绍。我们先来关注推导过程中得到的一些有趣的形式，前面得到$w=\sum_{i=1}^{n}\alpha_{i}y_{i}x_{i}$，因此

$$
\begin{align}
f(x)&=(\sum_{i=1}^{n}\alpha_{i}y_{i}x_{i})^{T}x+b\\
      &=\sum_{i=1}^{n}\alpha_{i}y_{i}\langle x_{i},x\rangle +b
\end{align}
$$

这里的形式的有趣之处在于，对于新点$x$的预测，只需要计算它与训练数据点的内积即可，这一点至关重要，是之后使用kernel进行非线性推广的基本前提。此外，所谓Supporting Vector也在这里显示出来--事实上，所有非SV所对应的系数$\alpha$都等于0，因此新点的内积计算实际上只要针对少量的支持向量而不是所有的训练数据。

为什么非支持向量对应的$\alpha$等于0？回忆我们之前得到的Lagrange目标函数：

$$
\max_{\alpha_{i}\geq 0}\mathcal{L}(w,b,\alpha)=\max_{\alpha_{i}\geq 0}\frac{1}{2}\|w\|^2-\sum_{i=1}^{n}\alpha_{i}\color{red} {(y_{i}(w^{T}x_{i}+b)-1)}
$$

注意到如果是支持向量的话，上式红色部分为0，而对于非支持向量，functional margin会大于1，因此红色部分大于0，而$\alpha_{i}$非负，为满足最大化，$\alpha_i$必须为0。

至此，我们得到了支持向量的定义和来由。

## Outliers
如果数据有噪音，如图所示

![image](https://github.com/ColdCodeCool/ColdCodeCool.github.io/raw/master/images/svmplot2.png)

对于这种偏离正常位置很远的数据点，我们称之为outlier，在我们原来的SVM模型里，outlier的存在有可能造成很大影响，因为超平面本身就是只有少数几个support vector组成的，如果这些support vector里又存在outlier的话，影响就很大。

用黑圈圈起来的那个蓝点是一个outlier，它偏离了自己原本所应该在的那个半空间，如果直接忽略掉的话，原来的分割超平面还是挺好的，但是由于这个outlier的出现，导致分割超平面不得不被挤歪了，变成图中黑色虚线所示，同时margin也相应变小了。当然，更严重的情况是，如果这个outlier再往右上移动一些距离的话，我们将无法构造出能将数据分开的超平面来。

为了处理这种情况，SVM允许数据点在一定程度上偏离一下超平面。例如上图中，黑色实线所对应的距离，就是该outlier偏离的距离，如果把它移动回来，就刚好落在原来的超平面上，而不会使得超平面发生变形了。具体来说，原来的约束条件

$$
y_{i}(w^{T}x_{i}+b)\geq 1，i=1，\ldots，n
$$

现在变成

$$
y_{i}(w^{T}x_{i}+b)\geq 1-\xi_i，i=1，\ldots，n
$$

其中$\xi_i\geq 0$成为松弛变量(slack variable)，对应数据点允许偏离的functional margin量。当然，如果允许$\xi_i$任意大的话，那任意的超平面都是符合条件的了。所以，我们在原来的目标函数后面加上一项，使得这些$\xi_i$的总和也要最小：

$$\min \frac{1}{2}\|w\|^2+\color{red} {+C\sum_{i=1}^{n}\xi_i}$$

其中$C$是一个参数，用于控制目标函数中两项之间的权重。其中，$\xi_{i}$是要优化的变量，而$C$是一个预先设定的常量。经过同样的Lagrange变换，得到dual problem为

$$\max_{\alpha}\sum_{i=1}^{n}\alpha_{i}-\frac{1}{2}\sum_{i,j=1}^{n}\alpha_{i}\alpha_{j}y_{i}y_{j}\langle x_i,x_j\rangle$$

$$
\begin{align*}
&\begin{array}{t}
s.t.，& 0\leq \alpha_{i}\leq C，i=1,\ldots，n\\
      & \sum_{i=1}^{n}\alpha_{i}y_i
\end{array}
\end{align*}
$$

至此，我们的SVM多了一项可以容忍噪音的技能。

## Kernels
前面介绍了线性情况下的SVM，它通过寻找一个线性的超平面来达到对数据进行分类的目的。不过，由于是线性方法，所以对非线性的数据就没有办法处理了。如图所示，

![image](https://github.com/ColdCodeCool/ColdCodeCool.github.io/raw/master/images/svmplot3.png)

该图显示的数据集的一个理想分界应该是一个圆圈。如果用$X_1$和$X_2$来表示这个二维平面的两个坐标的话，那一条二次曲线的方程可以写为

$$a_{1}X_{1}+a_{2}X_{1}^2+a_{3}X_2+a_{4}X_{2}^{2}+a_{5}X_{1}X_{2}+a_6=0$$

注意上面的形式，如果我们构造另外一个五维的空间，其中五个坐标的值分别为$Z_1=X_1$，$Z_2=X_{1}^{2}$，$Z_{3}=X_2$，$Z_4=X_{2}^2$，$Z_5=X_{1}X_{2}$，那么显然，上面方程在新的坐标系下可以写成

$$\sum_{i=1}^{5}a_{i}Z_{i}+a_6=0$$

关于新的坐标$Z$，这正是一个hyper plane的方程。也就是说，如果我们做一个映射$\phi:\mathbb{R}^2\rightarrow \mathbb{R}^5$，将$X$按照上面规则映射为$Z$，那么在新空间中原来的数据将变成线性可分的，从而使用之前推导出的线性分类算法就可以处理了。这正是kernel方法处理非线性问题的基本思想。

这样一来，求解的dual problem变成

$$\max_{\alpha}\sum_{i=1}^{n}\alpha_{i}-\frac{1}{2}\sum_{i,j=1}^{n}\alpha_{i}\alpha_{j}y_{i}y_{j}\langle \phi(x_i),\phi(x_j)\rangle$$

$$
\begin{align*}
&\begin{array}{t}
s.t.，& 0\leq \alpha_{i}\leq C，i=1,\ldots，n\\
      & \sum_{i=1}^{n}\alpha_{i}y_i
\end{array}
\end{align*}
$$

但是这样并没有解决问题，最初的例子中，我们对一个二维空间做映射，得到五个维度，如果原始空间是三维，那么得到19维，我们甚至可以得到无穷维，这样就根本无从计算。

因此我们考虑kernel，还从最简单的例子出发，设两个二维向量$x_{1}=(\eta_{1},\eta_2)^{T}$和$x_2=(\eta_{1},\eta_{2})^T$，而$\phi(.)$是前面说的五维空间的映射，因此映射过后的内积为：

$$\langle \phi(x_1),\phi(x_2)\rangle = \eta_{1}\xi_1+\eta_{1}^{2}\xi_{1}^{2}+\eta_{2}\xi_2+\eta_{2}^{2}\xi_{2}^2+\eta_{1}\eta_{2}\xi_{1}\xi_{2}$$

另外，我们注意到：

$$(\langle x_1+x_2\rangle+1)^2=2\eta_{1}\xi_1+\eta_{1}^{2}\xi_{1}^{2}+2\eta_{2}\xi_2+\eta_{2}^{2}\xi_{2}^2+2\eta_{1}\eta_{2}\xi_{1}\xi_2+1$$

二者有很多相似的地方，实际上我们只要把某几个维度线性放缩一下，然后加上常数维度，上面的式子就和高维映射后的内积结果相同。区别在什么地方呢？一个是映射到高维空间中，然后再根据内积的公式进行计算；而另一个则直接在原来的低维空间中进行计算，而不需要显式地写出映射后的结果。回忆刚才提到的映射维度爆炸，在前一种方法已经无法计算的情况下，后一种方法依旧能从容处理。

我们把这里的计算两个向量在映射过后的空间中的内积的函数叫做核函数(Kernel Function)，例如，在刚才的例子中，我们的核函数为

$$\kappa(x_1,x_2)=(\langle x_1,x_2\rangle +1)^2$$

核函数能简化映射空间中的内积运算--刚好碰巧的是，在我们的SVM里需要计算的地方数据向量总是以内积形式出现。最理想的情况下，我们希望知道数据的具体形状和分布，从而得到一个刚好可以将数据映射成线性可分的$\phi(.)$，然后通过这个$\phi(.)$得出对应的$k(..)$进行内积计算。然而第二步通常是非常困难甚至完全没法做的，不过由于第一步也是几乎无法做到，所以人们通常都是“胡乱”选择映射的。通常会从一些常用的核函数中选择例如：

- 多项式核$\kappa(x_1,x_2)=(\langle x_1,x_2\rangle + R)^d$，显然刚才我们举的例子是这里多项式核的一个特例。
- 高斯核$\kappa(x_1,x_2)=exp(-\frac{\|x_1-x_2\|^2}{2\sigma^2})$,又叫RBF（Radial Basis Function）核。这个核就是最开始提到过的会将原始空间映射为无穷维空间的。不过，如果$\sigma$选得很大的话，高次特征上的权重实际上衰减的非常快，所以实际上相当于一个低维的子空间；反过来，如果$\sigma$选得很小，则可以将任意的数据映射为线性可分--当然，这不一定是好事，因为随之而来的可能是非常严重的过拟合问题。不过，总的来说，通过调控参数$\sigma$，高斯核实际上具有相当高的灵活性，也是使用最广泛的核函数之一。
- 线性核$\kappa(x_1,x_2)=\langle x_1,x_2\rangle$，这实际上就是原始空间中的内积。

总结一下，对于非线性情况，SVM的处理方法是选择一个核函数，通过将数据映射到高维空间，来解决在原始空间中线性不可分的问题。除了SVM之外，任何将计算表示为数据点内积的方法，都可以使用核方法进行非线性扩展。

## Sequential Minimal Optimization



