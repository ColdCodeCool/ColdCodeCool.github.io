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
\begin{equation}
f(x_i)=\sum_{n=1}^{p}w_{n}x_{in} + w_0=w^{T}x_i
\end{equation}
其中w_0成为bias，上式通过增加$x_{i0}=1$把$w_0$也吸收到向量表达中了，上式$x_i$有p+1维。

线性回归的目标是用预测结果尽可能地拟合目标label，用最常见的Least Square作为loss function：
\begin{equation}
J(w)=\frac{1}{n}\sum_{i=1}^n(y_i-f(x_i))^2=\frac{1}{n}\lVert y - Xw \rVert^2
\end{equation}
可以直接求出最优解:
\begin{equation}
w^{*}=(X^{T}X)^{-1}X^{T}y
\end{equation}
直接法求解的问题在于上述协方差矩阵$X^{T}X$不可逆时，目标函数最小化必要条件求解方程时有无穷解，无法求出最优解。特别地，当$p>n$时，必然导致上述情况，并且此时也将存在overfitting的问题。此时需要对$w$做一些限制，使得它的最优解空间变小，也就是所谓的regularization。

## Ridge regression
最常见的限制是对$w$的模做约束，如ridge regression,岭回归，也即在线性回归的基础上加上$l_{2}-norm$的约束，loss function变成:
\begin{equation}
J_{R}(w)=\frac{1}{2}\lVert y - Xw \rVert^2 + \frac{\lambda}{2}\lVert w \rVert^2
\end{equation}
此时有解析解:
\begin{equation}
\hat{W_R}=(X^{T}X+\lambda I)^{-1}X^{T}y
\end{equation}
其中$\lambda>0$是一个参数，加入正则项之后解就有了一些很好的性质，首先是对$w$的模做约束，使得它的数值会比较小，很大程度上减轻了overfitting的问题，其次是上面的求逆部分肯定可逆，在实际中使用ridge regression通过调节$\lambda$可以得到不同的回归模型。

实际上ridge regression可以用下面的优化目标形式表达：
\begin{aligned}
& \min_{w} \quad \frac{1}{2} \lVert y - Xw \rVert^2， \\
& s.t. \lVert w \rVert_{2} < \theta \\
\end{aligned}
也就是说，我们依然可以优化线性回归的目标，但是条件是$w$的模长不能超过限制$\theta$。上面两种优化形式是等价的，可以找到一一对应的$\lambda$和$\theta$。

### 稀疏约束，LASSO
几种范数定义
0-范数：
\begin{equation}
\lVert w \rVert_{0}=\sum_{i}1(w_i\neq 0)
\end{equation}
1-范数：
\begin{equation}
\lVert w \rVert_{1}=\sum_{1}|w_i|
\end{equation}
2-范数：
\begin{equation}
\lVert w \rVert_{2}=(\sum_{i}w_{i}^{2})^{\frac{1}{2}}
\end{equation}
$\infty$-范数：
\begin{equation}
\lVert w \rVert_{\infty}=max(|w_1|,|w_2|,\cdots,|w_n|)
\end{equation}
如前面的ridge regression，对w做2范数约束，就是把解约束在一个$l_{2}-ball$里面，放缩是对球的半径放缩，因此w的每一个维度都在以同一个系数放缩，这样的放缩不会产生稀疏的解，即某些$w$的维度是0。而实际应用中，数据的维度中是存在噪音和冗余的，稀疏的解可以找到有用的维度并且减少冗余，提高回归预测的准确性和鲁棒性(减少了overfitting)。在压缩感知，稀疏编码等非常多的机器学习模型中都需要用到稀疏约束。

稀疏约束最直观的形式应该是约束0范数，如上面的范数介绍，$w$的0范数是求$w$中非零元素的个数。如果约束$\lVert w \rVert_{0} \leq k$，就是约束非零元素个数不大于k。不过很明显，0范数是不连续的且非凸的，如果在线性回归中加上0范数的约束，就变成了一个组合优化问题:挑出$\leq$ k个系数然后做回归，找到目标函数的最小值对应的系数组合，是一个NP问题。

值得注意的是，1范数也可以达到稀疏的效果，是0范数的最优凸近似，在一定条件下，0范数与1范数以概率1意义下等价。很重要的一点是，1范数容易求解，并且是凸的，所以几乎看得到稀疏约束的地方都是用的1范数。

回到线性回归的讨论，就引出了LASSO(Least Absolute Shrinkage and Selection Operator)的问题：
\begin{aligned}
& \min_{w} \quad \frac{1}{2} \lVert y - Xw \rVert^2， \\\\
& s.t. \lVert w \rVert_{1} < \theta \\\\
\end{aligned}
也就是说约束在一个$l_{1}-ball$里面。ridge和lasso的效果如图：![](https://github.com/ColdCodeCool/ColdCodeCool.github.io/raw/master/images/Selection_028.png)红色的椭圆和蓝色区域的切点就是目标函数的最优解，我们可以看到，如果是圆，则很容易切到圆周的任意一点，但很难切到坐标轴上，因此没有稀疏；如果是菱形或多边形，则很容易切到坐标轴上，因此很容易产生稀疏的结果。这也说明了为什么1范数是稀疏的。

### LASSO稀疏性的进一步理解：
类似Ridge，我们也可以写出LASSO的优化目标函数：
\begin{equation}
J_{L}(w)=\frac{1}{2}\lVert y-Xw \rVert^2+\lambda \sum_{i}|w_i|
\end{equation}
根据一般的思路，我们希望对$J_{L}(w)$求导数=0求出最优解，即$\nabla J_{L}(w)=0$,但是1范数在0点是连续不可导的，没有gradient，这个时候需要subgradient：

*定义1：记$f$:$U\rightarrow R$是一个定义在欧式空间凸集$R^{n}$上的实凸函数，在该空间中的一个向量$v$称为$f$在$x_0\in U$的次梯度(subgradient),如果对于任意$x\in U$,满足$f(x)-f(x_0)\geq v(x-x_0)$成立。*

由在点$x_0$处的所有subgradient所组成的集合称为$x_0$处的subdifferential，记为$\partial f(x_0)$。注意subgradient和subdifferential只是对凸函数定义的。例如一维的情况，$f(x)=|x|$，在x=0处的subdifferential就是[-1,1]这个区间。又例如下图中:![](https://github.com/ColdCodeCool/ColdCodeCool.github.io/raw/master/images/Selection_029.png)

在$x_0$不同红线的斜率就是表示subgradient的大小，有无穷多。
注意在$x$的gradient存在的点，subdifferential将是由gradient构成的一个单点集合。这样就将gradient的概念加以推广了。这个推广有一个很好的性质(condition for global minimizer)。

性质1：点$x\_0$是凸函数$f$的全局最小值，当且仅当0$\in \partial f(x\_0)$。

为了方便说明，需要做一个简化假设，即数据$X$的列向量是orthonomal的，即$X^{T}X=I$(当然没有这个假设LASSO也可以运作)。那么线性回归的最优解是:
\begin{equation}
w^{*}=X^{T}y
\end{equation}
假设lasso问题$J_{L}(w)$的全局最优解是$\bar{w}\in R^{n}$,考察它的任意一个维度$\bar{w}^j$，需要分别讨论两种情况：

情况1：gradient存在的区间，即$\bar{w}^j\neq 0$。

由于gradient在最小值点x=0，因此有:
\begin{equation}
\frac{\partial{J_{L}(w)}}{\partial{w^j}}\bigg \rvert\_{\bar{w}^j}=0
\end{equation}
。所以有:
\begin{align}
-(X^{T}y-X^{T}X \bar{w})_{j}+\lambda sgn(\bar{w}^j)=0
\end{align}
其中$\lambda \geq 0$。所以:

$$ \bar{w}^j=w^{*j}-\lambda sgn(\bar{w}^j)=sgn(w^{*j})(|w^{*j}|-\lambda) $$

$$ (|w^{*j}|-\lambda)=|\bar{w}^j|\neq 0 $$

从而有:

$$ \bar{w}^j=sgn(w^{*j})(|w^{*j}|-\lambda)_{+} $$

,其中$(x)_{+}$表示取$x$的正数部分,$(x)_{+}=\max(x,0)$。

情况2:gradient不存在，即$\bar{w}^j$=0

根据前面的性质1，如果$\bar{w}^j$是最小值，则:
\begin{equation}
0\in \partial{J\_{L}(\bar{w})}=-(X^{T}y-X^{T}X\bar{w})+\lambda e=\bar{w}-w^{*}+\lambda e
\end{equation}

其中$e$是一个向量，每一个元素$e^j\in [-1,1]$,使得$$0=-w^{*j*}+\lambda e^j$$成立。因此:

\begin{align}
|w^{*j}|=\lambda |e^j| \leq \lambda
\end{align}

所以情况1和2可以合并。在这种特殊的orthonomal情况下，我们可以直接写出LASSO的最优解：

$$ \bar{w}^j=sgn(w^{*j})(|w^{*j}|-\lambda)_{+} $$

回顾ridge regression，若同样考虑orthonomal，则有:

\begin{equation}
\hat{w}\_{R}=\frac{1}{1+\lambda}w^*
\end{equation}
很容易得出结论，ridge实际上是做了一个放缩，而lasso实际是做了一个soft thresholding，把很多权重项置0了，所以就得到了稀疏的结果。

除了做回归，LASSO的稀疏结果天然可以做机器学习中的特征选择，把非零的系数对应的维度选出即可，达到对问题的精简，去噪，以及减轻overfitting。