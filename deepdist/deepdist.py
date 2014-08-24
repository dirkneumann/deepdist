import copy
import cPickle as pickle
from flask import Flask, request
from rwlock import RWLock
import socket
from threading import Thread
import urllib2

"""Lightning-Fast Deep Learning on Spark
"""
class DeepDist:
    def __init__(self, model, host='127.0.0.1:5000'):
        """DeepDist - Distributed deep learning.
        :param model: provide a model that can be trained in parallel on the workers
        """
        self.model  = model
        self.lock   = RWLock()
        self.descent  = lambda model, gradient: model
        self.host     = host
        self.state    = 'serving'
        self.served   = 0
        self.received = 0

    def __enter__(self):
        Thread(target=self.start).start()
        return self
    
    def __exit__(self, type, value, traceback):
        pass # need to shut down server here
        
    def start(self):
        app = Flask(__name__)

        @app.route('/')
        def index():
            return 'DeepDist'

        @app.route('/model', methods=['GET', 'POST', 'PUT'])
        def model_flask():
            i = 0
            while (self.state != 'serving') and (i < 20):
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
        
        self.descent = descent
        
        host = self.host   # will be pickled by rdd.mapPartitions
        
        def mapPartitions(data):
            return (send_gradient(gradient(fetch_model(host=host), data), host=host))
        
        return rdd.mapPartitions(mapPartitions).collect()

def fetch_model(host='localhost:5000'):
    request = urllib2.Request('http://%s/model' % host,
        headers={'Content-Type': 'application/deepdist'})
    return pickle.loads(urllib2.urlopen(request).read())

def send_gradient(gradient, host='localhost:5000'):
    if not gradient:
          return 'EMPTY'
    request = urllib2.Request('http://%s/update' % host, pickle.dumps(gradient, -1),
        headers={'Content-Type': 'application/deepdist'})
    return urllib2.urlopen(request).read()
