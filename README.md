Training deep belief networks requires extensive data and computation. [DeepDist](http://deepdist.com) accelerates the training by distributing stochastic gradient descent for data stored on HDFS / Spark via a simple Python interface. Overview: [deepdist.com](http://deepdist.com)

Quick start:
----

Training of [word2vec](https://code.google.com/p/word2vec/) model on [wikipedia](http://dumps.wikimedia.org/enwiki/) in 15 lines of code:

```python
from deepdist import DeepDist
from gensim.models.word2vec import Word2Vec
from pyspark import SparkContext

sc = SparkContext()
corpus = sc.textFile('enwiki').map(lambda s: s.split())

def gradient(model, sentences):  # executes on workers
    syn0, syn1 = model.syn0.copy(), model.syn1.copy()
    model.train(sentences)
    return {'syn0': model.syn0 - syn01, 'syn1': model.syn1 - syn1}

def descent(model, update):      # executes on master
    model.syn0 += update['syn0']
    model.syn1 += update['syn1']

with DeepDist(Word2Vec(corpus.collect()) as dd:

    dd.train(corpus, gradient, descent)
    print dd.model.most_similar(positive=['woman', 'king'], negative=['man'])
```

How does it work?
----

DeepDist implements a [Sandblaster](http://research.google.com/archive/large_deep_networks_nips2012.html)-like stochastic gradient descent. It start a master model server (on port 5000). On each data node, DeepDist fetches the model from the server, and then calls gradient(). After computing the gradient for each RDD partition, gradient updates are send the the server. On the server, the master model is then updated by descent().

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