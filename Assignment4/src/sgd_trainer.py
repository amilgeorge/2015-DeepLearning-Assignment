'''
Created on 16-May-2015

@author: amilgeorge
'''
from DataLoader import DataLoader
import numpy as np

import theano
import theano.tensor as T

class sgd_trainer(object):
    '''
    classdocs
    '''


    def __init__(self,batch_size=20,learning_rate=0.01, sparsity_lambda=0.01, n_epochs=20,reconstruction_cost="cross_entropy", sparsity_cost = "kl"):
        '''
        Constructor
        '''
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = n_epochs
        self.sparsity_lambda = sparsity_lambda
        self.reconst_cost = reconstruction_cost
        self.sparsity_cost = sparsity_cost
    
    def trainAutoEncoder(self,ae):    
        
        data_loader = DataLoader()
        datasets=data_loader.load_shared_data()
        train_set_x, train_set_y = datasets[0]
        
        n_train_batches = train_set_x.get_value(borrow=True).shape[0] / self.batch_size
        
        index = T.lscalar()   
        x = ae.input
        
        
        main_cost = ae.get_cross_enropy_cost()
        
        if self.reconst_cost == "sqr":
            main_cost =ae.get_squared_error_cost()
        
        sparsity_cost= ae.get_KL_divergence_cost()
        
        if self.sparsity_cost =="l1":
            sparsity_cost = ae.get_L1_cost()
        
        cost = main_cost + self.sparsity_lambda * sparsity_cost
        
        gparams = T.grad(cost, ae.params)
        
        updates = [
            (param, param - self.learning_rate * gparam)
            for param, gparam in zip(ae.params, gparams)
        ]
        
        train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * self.batch_size: (index + 1) * self.batch_size]
        }
        )
        
        for epoch_no in xrange(self.num_epochs):
            c=[]
            for batch_index in xrange(n_train_batches):
                c.append(train_model(batch_index))
            
            print 'Training epoch no: %d, cost ' % epoch_no, np.mean(c)
            
        