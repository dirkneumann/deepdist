from deepdist import DeepDist
from gensim.models.word2vec import Vocab, Word2Vec
import numpy as np
import os
from pyspark import SparkConf, SparkContext
import urlparse
import random
import sys
import time

conf = SparkConf()
sc = SparkContext(conf=conf)

corpus = sc.textFile('s3n://dd-enwiki/*a.txt.gz').map(lambda s: s.split()).filter(lambda s: len(s) > 0)

print 'Build vocabulary...'
s = corpus   \
    .flatMap(lambda s: [(w, 1) for w in s])   \
    .reduceByKey(lambda a, b: a+b)            \
    .filter(lambda x: x[1] >= 5)              \
    .map(lambda x: (x[1], x[0]))              \
    .collect()
    #.sortByKey(False)                         \
    #.collect()

vocab = {}
for i, (c, w) in enumerate(s):
    if i >= 1000000:
        break
    if (i + 1) % 100000 == 0:
        print i+1
    vocab[w] = Vocab(count=c)

def build_vocab(model, vocab):
    model.word_count  = long(0)
    model.total_words = long(0)
    model.vocab, model.index2word = {}, []
    for word, v in vocab.iteritems():
        if v.count >= model.min_count:
            v.index = len(model.vocab)
            model.index2word.append(word)
            model.vocab[word] = v
        model.total_words += v.count
    print "total %i word types after removing those with count<%s" % (len(model.vocab), model.min_count)
    
    if model.hs:
        model.create_binary_tree()
    if model.negative:
        model.make_table()
    
    model.precalc_sampling()
    model.reset_weights()

model = Word2Vec()
build_vocab(model, vocab)

model.version = 1

model.ssyn0 = 0   # AdaGrad: sum of squared gradients
model.ssyn1 = 0

'''
print 'Pretrain model...'
for filename in os.listdir('enwiki')[:10]:
    model.train([s.split() for s in open('enwiki/%s' % filename)])
'''

def gradient(model, data):
    syn0, syn1 = model.syn0.copy(), model.syn1.copy()
    words  = model.train(data, word_count=model.word_count, total_words=model.total_words)
    update = {
        'syn0': model.syn0 - syn0, 
        'syn1': model.syn1 - syn1, 
        'words': words - model.word_count
    }
    return update

t = time.time()

def descent(model, update):
    alpha = max(model.min_alpha, model.alpha * (1.0 - 1.0 * model.word_count / model.total_words))
    
    syn0 = update['syn0'] / alpha
    syn1 = update['syn1'] / alpha
    
    model.ssyn0 += syn0 * syn0
    model.ssyn1 += syn1 * syn1
    
    alpha0 = alpha / (1e-6 + np.sqrt(model.ssyn0))
    alpha1 = alpha / (1e-6 + np.sqrt(model.ssyn1))
    
    model.syn0 += syn0 * alpha0
    model.syn1 += syn1 * alpha1
    
    model.word_count = long(model.word_count) + long(update['words'])
    model.version += 1

    if random.random() < 0.1:
        print 'Evaluating model...'
        log = open('train.txt', 'a')
        try:
            del model.syn0norm
        except:
            pass
        for row in model.accuracy('questions-words.txt'):
            if row['section'] != 'total':
                continue
            print >>log, (' %i %.1f%% v%i %.1f %.1f' %
                (row['correct'], 100.0 * row['correct'] / (row['incorrect'] + row['correct']),
                 model.version, 1.0 * row['correct'] / model.version, time.time() - t))


print 'Train model...'
with DeepDist(model, batch=100000, master='54.188.71.71:5000') as dd:
    
    dd.train(corpus.sample(False, 0.01), gradient, descent)

print 'Saving model to "model.bin"...'
model.save_word2vec_format('model.bin', binary=True)

print 'Evaluate model...'
del model.syn0norm
for row in model.accuracy('questions-words.txt'):
    if row['section'] != 'total':
        continue
    print(' %i %.1f%% v%i %.1f' % 
        (row['correct'], 100.0 * row['correct'] / (row['incorrect'] + row['correct']), 
         model.version, 1.0 * row['correct'] / model.version))

print model.most_similar(positive=['woman', 'king'], negative=['man'])
