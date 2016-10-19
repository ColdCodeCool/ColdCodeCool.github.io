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
x_t: 8000\\
o_t: 8000\\
s_t: 100\\
U: 100 X 8000\\
V: 8000 X 100\\
W: 100 X 100\\
$$

从上面看出我们将要学习$2HC + H*H$个参数, 但$x_t$是一个one-hot向量,
将其与$U$想乘, 则仅仅是等效于取$U$的某一列. 所以我们的瓶颈在于$Vs_t$, 而这
也是为什么我们要保持一个较小的vocabulary size.

{% highlight python %}

import numpy as np
import nltk
import csv
import itertools

vocabulary_size = 8000
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"
 
# Read the data and append SENTENCE_START and SENTENCE_END tokens
print "Reading CSV file..."
with open('data/reddit-comments-2015-08.csv', 'rb') as f:
    reader = csv.reader(f, skipinitialspace=True)
    reader.next()
    # Split full comments into sentences
    sentences = itertools.chain(*[nltk.sent_tokenize(x[0].decode('utf-8').lower()) for x in reader])
    # Append SENTENCE_START and SENTENCE_END
    sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]
print "Parsed %d sentences." % (len(sentences))
     
# Tokenize the sentences into words
tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]
 
# Count the word frequencies
word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
print "Found %d unique words tokens." % len(word_freq.items())
 
# Get the most common words and build index_to_word and word_to_index vectors
vocab = word_freq.most_common(vocabulary_size-1)
index_to_word = [x[0] for x in vocab]
index_to_word.append(unknown_token)
word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])
 
print "Using vocabulary size %d." % vocabulary_size
print "The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1])
 
# Replace all words not in our vocabulary with the unknown token
for i, sent in enumerate(tokenized_sentences):
    tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]
 
print "\nExample sentence: '%s'" % sentences[0]
print "\nExample sentence after Pre-processing: '%s'" % tokenized_sentences[0]
 
# Create the training data
X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])

class RNNNumpy:
	def __init__(self, word_dim, hidden_dim=100, bptt_truncate=4):
		# Assign instance variables
		self.word_dim = word_dim
		self.hidden_dim = hidden_dim
		self.bptt_truncate = bptt_truncate
		# Randomly initilize the network parameters
		self.U = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
		self.V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim))
		self.W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))


	def forward_propagation(self, x):
		# The total number of time steps
		T = len(x)
		# During forward propagation we save all hidden states in s because need them later.
		# We add one additional element for the initial hidden, which we set to 0
		s = np.zeros((T+1, self.hidden_dim))
		s[-1] = np.zeros(self.hidden_dim)
		# The outputs at each time step. Again, we save them for later.
		o = np.zeros((T, self.word_dim))
		# For each time step...
		for t in np.arange(T):
			# Note that we are indexing U by x[t]. This is the same as multiplying U with a one-hot vector.
			s[t] = np.tanh(self.U[:, x[t]] + self.W.dot(s[t-1]))
			o[t] = self.softmax(self.V.dot(s[t]))
		return [o, s]

	def predict(self, x):
		# Perform forward propagation and return index of the highest score
		o, s = self.forward_propagation(x)
		return np.argmax(o, axis=1)

	def softmax(self, x):
		return np.exp(x) / np.exp(x).sum(axis=0)

	def calculate_total_loss(self, x, y):
		L = 0
		# For each sentence..
		for i in np.arange(len(y)):
			o, s = self.forward_propagation(x[i])
			# we only care about our prediction of the "correct" words
			correct_word_predictions = o[np.arange(len(y[i])), y[i]]
			# Add to the loss based on how off we were
			L += -1 * np.sum(np.log(correct_word_predictions))
		return L

	def calculate_loss(self, x, y):
		# Divide the total loss by the number of training examples
		N = np.sum((len(y_i) for y_i in y))
		return self.calculate_total_loss(x,y)/N

	def bptt(self, x, y):
		T = len(y)
		# Perform forward propagation
		o, s = self.forward_propagation(x)
		# We accumulate the gradients in these variables
		dLdU = np.zeros(self.U.shape)
		dLdV = np.zeros(self.V.shape)
		dLdW = np.zeros(self.W.shape)
		delta_o = o
		delta_o[np.arange(len(y)), y] -= 1.
		# For each output backwards...
		for t in np.arange(T)[::-1]:
		    dLdV += np.outer(delta_o[t], s[t].T)
		    # Initial delta calculation
		    delta_t = self.V.T.dot(delta_o[t]) * (1 - (s[t] ** 2))
		    # Backpropagation through time (for at most self.bptt_truncate steps)
		    for bptt_step in np.arange(max(0, t-self.bptt_truncate), t+1)[::-1]:
		        # print "Backpropagation step t=%d bptt step=%d " % (t, bptt_step)
		        dLdW += np.outer(delta_t, s[bptt_step-1])              
		        dLdU[:,x[bptt_step]] += delta_t
		        # Update delta for next step
		        delta_t = self.W.T.dot(delta_t) * (1 - s[bptt_step-1] ** 2)
		return [dLdU, dLdV, dLdW]

	def gradient_check(self, x, y, h=0.001, error_threshold=0.01):
		# Calculate the gradients using backpropagation. We want to checker if these are correct.
		bptt_gradients = self.bptt(x, y)
		# List of all parameters we want to check.
		model_parameters = ['U', 'V', 'W']
		# Gradient check for each parameter
		for pidx, pname in enumerate(model_parameters):
		# Get the actual parameter value from the mode, e.g. model.W
		parameter = operator.attrgetter(pname)(self)
		print "Performing gradient check for parameter %s with size %d." % (pname, np.prod(parameter.shape))
		# Iterate over each element of the parameter matrix, e.g. (0,0), (0,1), ...
		it = np.nditer(parameter, flags=['multi_index'], op_flags=['readwrite'])
		while not it.finished:
		    ix = it.multi_index
		    # Save the original value so we can reset it later
		    original_value = parameter[ix]
		    # Estimate the gradient using (f(x+h) - f(x-h))/(2*h)
		    parameter[ix] = original_value + h
		    gradplus = self.calculate_total_loss([x],[y])
		    parameter[ix] = original_value - h
		    gradminus = self.calculate_total_loss([x],[y])
		    estimated_gradient = (gradplus - gradminus)/(2*h)
		    # Reset parameter to original value
		    parameter[ix] = original_value
		    # The gradient for this parameter calculated using backpropagation
		    backprop_gradient = bptt_gradients[pidx][ix]
		    # calculate The relative error: (|x - y|/(|x| + |y|))
		    relative_error = np.abs(backprop_gradient - estimated_gradient)/(np.abs(backprop_gradient) + np.abs(estimated_gradient))
		    # If the error is to large fail the gradient check
		    if relative_error &gt; error_threshold:
		        print "Gradient Check ERROR: parameter=%s ix=%s" % (pname, ix)
		        print "+h Loss: %f" % gradplus
		        print "-h Loss: %f" % gradminus
		        print "Estimated_gradient: %f" % estimated_gradient
		        print "Backpropagation gradient: %f" % backprop_gradient
		        print "Relative Error: %f" % relative_error
		        return
		    it.iternext()
		print "Gradient check for parameter %s passed." % (pname)

	def numpy_sdg_step(self, x, y, learning_rate):
		# Calculate the gradients
		dLdU, dLdV, dLdW = self.bptt(x, y)
		# Change parameters according to gradients and learning rate
		self.U -= learning_rate * dLdU
		self.V -= learning_rate * dLdV
		self.W -= learning_rate * dLdW

# Outer SGD Loop
# - model: The RNN model instance
# - X_train: The training data set
# - y_train: The training data labels
# - learning_rate: Initial learning rate for SGD
# - nepoch: Number of times to iterate through the complete dataset
# - evaluate_loss_after: Evaluate the loss after this many epochs
def train_with_sgd(model, X_train, y_train, learning_rate=0.005, nepoch=100, evaluate_loss_after=5):
	# We keep track of the losses so we can plot them later
	losses = []
	num_examples_seen = 0
	for epoch in range(nepoch):
	    # Optionally evaluate the loss
	    if (epoch % evaluate_loss_after == 0):
	        loss = model.calculate_loss(X_train, y_train)
	        losses.append((num_examples_seen, loss))
	        time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
	        print "%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss)
	        # Adjust the learning rate if loss increases
	        if (len(losses) &gt; 1 and losses[-1][1] &gt; losses[-2][1]):
	            learning_rate = learning_rate * 0.5 
	            print "Setting learning rate to %f" % learning_rate
	        sys.stdout.flush()
	    # For each training example...
	    for i in range(len(y_train)):
	        # One SGD step
	        model.sgd_step(X_train[i], y_train[i], learning_rate)
	        num_examples_seen += 1

{% endhighlight %}
