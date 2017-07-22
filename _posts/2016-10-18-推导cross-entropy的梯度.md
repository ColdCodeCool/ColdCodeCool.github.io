---
layout: post
title: "Derivation of the gradient of the Cross-Entropy loss function"
date: 2016-10-18 10:21:16
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
\begin{equation}
\begin{split}
E(\theta) & =-\log(\hat{y_k})\\
		  & =-\log(\frac{exp(\theta_k)}{\sum_{j}exp(\theta_j)})\\
		  & =-\theta_k + \log(\sum_{j}exp(\theta_j))\\
\end{split}
\end{equation}
$$

从而有:

$$\frac{\partial E}{\partial \theta}=-\frac{\partial \theta_k}{\partial \theta} + \frac{\partial}{\partial \theta}(\log(\sum_{j}exp(\theta_j)))$$

又有$\frac{\partial \theta_k}{\partial \theta_k}=1$并且$\frac{\partial \theta_k}{\partial \theta_q}=0$ for $q \neq k$.则:

$$\frac{\partial \theta_k}{\partial \theta}=y$$

对于第二部分:

$$\frac{\partial}{\partial \theta_i}\log(\sum_{j}exp(\theta_j))=\frac{exp(\theta_i)}{\sum_{j}exp(\theta_j)}=\hat{y_i}$$

因此:

$$\frac{\partial E}{\partial \theta}=\frac{\partial}{\partial \theta}\log(\sum_{j}exp(\theta_j))-\frac{\partial \theta_k}{\partial \theta}=\hat{y}-y$$