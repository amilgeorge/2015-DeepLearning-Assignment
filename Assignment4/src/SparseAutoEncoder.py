'''
Created on 15-May-2015

@author: amilgeorge
'''


import numpy as np

import theano
import theano.tensor as T

class SparseAutoEncoder(object):
    '''
    classdocs
    '''


    def __init__(self, num_input,num_hidden,activation = T.nnet.sigmoid):
        '''
        Constructor
        '''
        self.input = T.matrix(name='input')
        
        self.activation = activation
        numpy_rng = np.random.RandomState(321)
        W1_init = np.asarray(
                numpy_rng.uniform(
                    low=-4 * np.sqrt(6. / (num_hidden + num_input)),
                    high=4 * np.sqrt(6. / (num_hidden + num_input)),
                    size=(num_input, num_hidden)
                ),
                dtype=theano.config.floatX  # @UndefinedVariable
            )
        self.W1 = theano.shared(value=W1_init, name='W', borrow=True)
        
        self.b1 = theano.shared(
                value=np.zeros(
                    num_hidden,
                    dtype=theano.config.floatX  # @UndefinedVariable
                ),
                name='b1',
                borrow=True
            )
        
        self.W2 = self.W1.T
        
        self.b2 = theano.shared(
                value=np.zeros(
                    num_input,
                    dtype=theano.config.floatX  # @UndefinedVariable
                ),
                name='b2',
                borrow=True
            )
        
        self.params = [self.W1, self.b1, self.b2]   
        
        hidden_vals = self.get_hidden_values(self.input)
        re_inp = self.get_reconstructed_input(hidden_vals)
        
        self.encoder = theano.function(
        [self.input],
        re_inp
        )
        
    def get_hidden_values(self, inpt):

        return self.activation(T.dot(inpt, self.W1) + self.b1)

    def get_reconstructed_input(self, hidden):
    
        return self.activation(T.dot(hidden, self.W2) + self.b2)
    
    def get_cross_enropy_cost(self):
        
        x=self.input
        y = self.get_hidden_values(x)
        z= self.get_reconstructed_input(y)
        L = - T.sum(x * T.log(z) + (1 - x) * T.log(1 - z), axis=1)
        cost = T.mean(L);
        
        return cost
     
    '''
    x = 20 X 728
    y = 20 X 500
    
    '''  
    def get_KL_divergence_cost(self):
        x=self.input
        y = self.get_hidden_values(x)
        
        rho = 0.05
        rho_cap = y
        KL =T.sum(rho * T.log(rho/rho_cap) + (1 - rho) * T.log((1-rho)/(1-rho_cap)),axis=1)
        
        cost = T.mean(KL)
        
        return cost 
    
    def encode(self,inpt):
        out=self.encoder(inpt)
        return out
            