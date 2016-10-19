---
layout: post
title: "Implementing RNN with python and numpy"
data: 2016-10-19 13:11:59
categories: TimeSeries
---
## A full RNN for language modeling
我们的目标是用RNN来建立一个语言模型. 我们有一个包含m个单词的句子,一个语言模型能使我们预测这个句子出现的概率:

$$P(w_1,...,w_m) = \prod_{i=1}^{m}P(w_i|w_1,...,w_i-1)$$

上式表明, 一个句子出现的概率是句子中所有单词在给定它之前单词出现的条件概率之积. 举个例子, 一个句子"He went to buy some chocolate"出现的概率是给定"He went to buy some"出现"chocolate"的概率, 及给定"He went to buy"出现"some"的概率等等之积. 为什么这样是可行的?为什么我们想要赋予一个被观察的句子概率? 首先, 这样一个模型可以用做打分机制. 比如, 一个机器翻译系统针对一个输入句子会给出许多候选翻译句子. 这时你就可以利用language model来选择最可能的句子.直觉上, 最可能的句子是语法正确的. 但是解决这种language model问题也会造成一种好的副作用, 因为我们可以根据前面的单词预测下一个单词, 这样的话我们就能够生成新的内容. 这便是生成模型.

### 数据预处理
训练语言模型需要文本输入, 我们可以从公开数据集来获取.

- tokenize text 使用NLTK的word_tokenize和sent_tokenize
- remove infrequent words (去除出现次数少的单词)将单词本的词汇量定位vocabulary_size, 在此之外的单词都记作unknown_token
- 给一个sentence加特殊的开头和结尾. 如以SENTENCE_START作为开头, SENTENCE_END作为结尾,将单词映射为向量之后, 可以由0来表示开头, 1来表示结尾. 举例,一个训练样本x可以表示为[0,179,341,416],其相应的标签y则应该是[179,341,416,1]. 我们的目标是预测下一个单词, 所以y就是左移一位之后最后一 位用1结尾.也就是说, 对单词179的正确预测就是他在x中的下一个单词.
- 重新构造输入数据. 我们不能简单地使用word index来作为输入数据, 取而代之, 我们可以将每一个单词表示成一个维度为vocabulary_size的one-hot vector. 这样每个单词都是一个向量, 那么训练数据x将是一个矩阵, 每一行代表一个单词. 我们的神经网络输出具有相似的格式. 每一个$o_t$都是一个vocabulary_size维的向量, 向量每一个元素代表了对应位置
单词成为下一个单词的概率.

我们可以概括一下下面两个方程:

$$
\begin{equation}
\begin{split}
s_t &= tanh(Ux_t + Ws_t-1)\\
o_t &= softmax(Vs_t)\\
\end{split}
\end{equation}
$$
深入研究之前, 一个好习惯是写下来每一个矩阵或向量的维度(预先定义vocabulary_size C=8000, hidden layer size H=100):

$$
x_t: 8000
o_t: 8000
s_t: 100
U: 100 X 8000
V: 8000 X 100
W: 100 X 100
$$
从上面看出我们将要学习2HC + H*H个参数, 但x_t是一个one-hot向量,
将其与U想乘, 则仅仅是等效于取U的某一列. 所以我们的瓶颈在于Vs_t, 而这
也是为什么我们要保持一个较小的vocabulary size.
