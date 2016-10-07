Training deep belief networks requires extensive data and computation. [DeepDist](http://deepdist.com) accelerates the training by distributing stochastic gradient descent for data stored on HDFS / Spark via a simple Python interface. Overview: [deepdist.com](http://deepdist.com)

Quick start:
----

Training of a [word2vec](https://code.google.com/p/word2vec/) model on [wikipedia](http://dumps.wikimedia.org/enwiki/) in 15 lines of code:

```python
from deepdist import DeepDist
from gensim.models.word2vec import Word2Vec
from pyspark import SparkContext

sc = SparkContext()
corpus = sc.textFile('enwiki').map(lambda s: s.split())

def gradient(model, sentences):  # executes on workers
    syn0, syn1 = model.syn0.copy(), model.syn1.copy()
    model.train(sentences)
    return {'syn0': model.syn0 - syn0, 'syn1': model.syn1 - syn1}

def descent(model, update):      # executes on master
    model.syn0 += update['syn0']
    model.syn1 += update['syn1']

with DeepDist(Word2Vec(corpus.collect())) as dd:

    dd.train(corpus, gradient, descent)
    print dd.model.most_similar(positive=['woman', 'king'], negative=['man'])
```

How does it work?
----

DeepDist implements a [Downpour](http://research.google.com/archive/large_deep_networks_nips2012.html)-like stochastic gradient descent. It start a master model server (on port 5000). On each data node, DeepDist fetches the model from the server, and then calls __gradient()__. After computing the gradient for each RDD partition, gradient updates are sent to the server. On the server, the master model is then updated by __descent()__.

![Alt text](http://deepdist.com/images/deepdistdesign.png)

Python module
----

[DeepDist](http://deepdist.com) provides a simple Python interface. The with statement starts the model server. Distributed gradient updates are computed on partitions of a resilient distributed dataset (RDD) data. The gradient updates are incorporated into the master model via custom descent method.

```python
from deepdist import DeepDist
 
with DeepDist(model) as dd:    # initialized server with any model    
    
    dd.train(data, gradient, descent)
    # train with an RDD "data" by computing distributed gradients and
    # descending the model parameters space according to gradient updates
 
def gradient(model, data):
    # model is a copy of the master model
    # data is an iterator for the current partition of the data RDD
    # returns the gradient update
 
def descent(model, update):
    # model is a reference to the server model
    # update is a copy of a worker's update
```

Training Speed
----

Training speed can be greatly enhanced by adaptively adjusting the learning rate by [AdaGrad](http://www.cs.berkeley.edu/~jduchi/projects/DuchiHaSi10.pdf). The complete Word2Vec model with 900 dimensions can be trained on the 19GB wikipedia corpus (using the words from the validation questions).

![Training](http://deepdist.com/images/training.png)

References
----

J Dean, GS Corrado, R Monga, K Chen, M Devin, QV Le, MZ Mao, M’A Ranzato, A Senior, P Tucker, K Yang, and AY Ng. [Large Scale Distributed Deep Networks](http://research.google.com/archive/large_deep_networks_nips2012.html). NIPS 2012: Neural Information Processing Systems, Lake Tahoe, Nevada, 2012.

T Mikolov, I Sutskever, K Chen, G Corrado, and J Dean. [Distributed Representations of Words and Phrases and their Compositionality](http://arxiv.org/pdf/1310.4546.pdf). In Proceedings of NIPS, 2013.

T Mikolov, K Chen, G Corrado, and J Dean. [Efficient Estimation of Word Representations in Vector Space](http://arxiv.org/pdf/1301.3781.pdf). In Proceedings of Workshop at ICLR, 2013.
