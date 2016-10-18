---
layout: post
title: "Derivation of the gradient of the Cross-Entropy loss function"
data: 2016-10-18 10:21:16
categories: Deeplearning
---
## 梯度推导
我们要证明:

$$
\frac{\partial E}{\partial \theta} = \hat{y} - y
$$

已知
$$
E(\theta) = -\sum_{i}y_i * \log(\hat{y_i})
$$
且
$$
\hat{y_i} = \frac{exp(\theta_i)}{\sum_{i}exp(\theta_{i})}
$$
又知道$y_j=0$ for $j\neq k$及$y_k=1$.所以:
$$
E(\theta) &= -\log(\hat{y_k}) \\
&= -\log(\frac{exp(\theta_k)}{\sum_{j}exp(\theta_j)})\\
&= -\theta_k + \log(\sum_{j}exp(\theta_j))\\
$$