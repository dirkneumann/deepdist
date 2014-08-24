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
