import copy
import cPickle as pickle
from multiprocessing import Process
from rwlock import RWLock
import socket
import sys
from threading import Thread
import urllib2
import urlparse

"""Lightning-Fast Deep Learning on Spark
"""
class DeepDist:
    def __init__(self, model, batch=None, master='127.0.0.1:5000'):
        """DeepDist - Distributed deep learning.
        :param model: provide a model that can be trained in parallel on the workers
        """
        self.model  = model
        self.lock   = RWLock()
        self.descent  = lambda model, gradient: model
        self.master   = master
        self.state    = 'serving'
        self.served   = 0
        self.received = 0
        self.batch    = batch
        self.server   = None

    def __enter__(self):
        Thread(target=self.start).start()
        # self.server = Process(target=self.start)
        # self.server.start()
        return self
    
    def __exit__(self, type, value, traceback):
        # self.server.terminate()
        pass # need to shut down server here
        
    def start(self):
        from flask import Flask, request

        app = Flask(__name__)

        @app.route('/')
        def index():
            return 'DeepDist'

        @app.route('/model', methods=['GET', 'POST', 'PUT'])
        def model_flask():
            i = 0
            while (self.state != 'serving') and (i < 1000):
                time.sleep(1)
                i += 1

            self.lock.acquire_read()
            self.served += 1
            model = copy.deepcopy(self.model)
            self.lock.release()
            
            return pickle.dumps(model, -1)
    

        @app.route('/update', methods=['GET', 'POST', 'PUT'])
        def update_flask():
            gradient = pickle.loads(request.data)

            self.lock.acquire_write()
            state = 'receiving'
            self.received += 1
            
            self.descent(self.model, gradient)
            
            if self.received >= self.served:
                self.received = 0
                self.served   = 0
                self.state    = 'serving'
            
            self.lock.release()
            return 'OK'
        
        print 'Listening to 0.0.0.0:5000...'
        app.run(host='0.0.0.0', debug=True, threaded=True, use_reloader=False)

    def train(self, rdd, gradient, descent):
        master = self.master   # will be pickled
        print 'master0: ', master
        if master == None:
            master = rdd.ctx._conf.get('spark.master')
        print 'master1: ', master
        if master.startswith('local['):
            master = 'localhost:5000'
        else:
            if master.startswith('spark://'):
                master = '%s:5000' % urlparse.urlparse(master).netloc.split(':')[0]
            else:
                master = '%s:5000' % master.split(':')[0]
        print '\n*** master: %s\n' % master

        self.descent = descent
        
        batch = self.batch
        
        def mapPartitions(data):
            last = 'dummy'
            class Iter:
              def __iter__(self):
                self.i = 0
                return self
              def next(self):
                if (batch == None) or (self.i < batch):
                  self.i += 1
                  last = data.next()
                  return last
                else:
                  return None
            res = []
            while last != None:
              res.append(send_gradient(gradient(fetch_model(master=master), Iter()), master=master))
            return res
        
        return rdd.mapPartitions(mapPartitions).collect()

def fetch_model(master='localhost:5000'):
    print '\n*** url: %s' % ('http://%s/model' % master)
    request = urllib2.Request('http://%s/model' % master,
        headers={'Content-Type': 'application/deepdist'})
    return pickle.loads(urllib2.urlopen(request).read())

def send_gradient(gradient, master='localhost:5000'):
    if not gradient:
          return 'EMPTY'
    request = urllib2.Request('http://%s/update' % master, pickle.dumps(gradient, -1),
        headers={'Content-Type': 'application/deepdist'})
    return urllib2.urlopen(request).read()
