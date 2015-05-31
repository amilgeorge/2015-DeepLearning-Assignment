'''
Created on 30-May-2015

@author: amilgeorge
'''

import os
import sys
import time

import numpy as np

import theano
import theano.tensor as T

#######################
#Rectified Linear Unit#
#######################
def relu(x):
    return T.switch(x<0, 0, x)

def dropout(rng,output,dropout_rate):
    srng = T.shared_randomstreams.RandomStreams(
            rng.randint(999999))
  
    mask = srng.binomial(n=1, p=1-dropout_rate, size=output.shape)
    
    output_dropout = output * T.cast(mask, theano.config.floatX)  # @UndefinedVariable
    return output_dropout


class NeuralNetwork(object):
    '''
    classdocs
    '''


    def __init__(self, num_input,num_hidden,num_output,activation='tanh',dropout_rate = 0.3):
        '''
        Constructor
        '''
        
        if activation == 'sigmoid':
            self.activation1 = T.nnet.sigmoid
        elif activation == 'relu':
            self.activation1 = relu
        else:
            self.activation1 = T.tanh
        
            
        
        self.input = T.matrix('x')
        
        
        rng = np.random.RandomState(1234)
        W1_values = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (num_input + num_hidden)),
                    high=np.sqrt(6. / (num_input + num_hidden)),
                    size=(num_input, num_hidden)
                ),
                dtype=theano.config.floatX  # @UndefinedVariable
            )
       
        if self.activation1 == T.nnet.sigmoid:
            W1_values *= 4
            
        self.W1 = theano.shared(value=W1_values, name='W1', borrow=True)
        
        b1_values = np.zeros((num_hidden,), dtype=theano.config.floatX)  # @UndefinedVariable
        
        self.b1 = theano.shared(value=b1_values, name='b1', borrow=True)
        
        self.output_hidden = self.activation1(T.dot(self.input, self.W1) + self.b1)
        
           
        
        self.output_hidden_dropout = dropout(rng, self.output_hidden, dropout_rate)
        
        self.W2 = theano.shared(
            value=np.zeros(
                (num_hidden, num_output),
                dtype=theano.config.floatX  # @UndefinedVariable
            ),
            name='W2',
            borrow=True
        )
        
        self.b2 = theano.shared(
            value=np.zeros(
                (num_output,),
                dtype=theano.config.floatX  # @UndefinedVariable
            ),
            name='b2',
            borrow=True
        )
        
      
        self.p_y_given_x = T.nnet.softmax(T.dot(self.output_hidden, self.W2) + self.b2)

        
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        
        self.L1 = (
            abs(self.W1).sum()
            + abs(self.W2).sum()
        )
        
        self.L2 = (
            (self.W1 ** 2).sum()
            + (self.W2 ** 2).sum()
        )
        
        self.params = [self.W1,self.b1,self.W2,self.b2]
        
    def negative_log_likelihood(self, y):

        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
    
    def negative_log_likelihood_dropout(self, y):
        p_y_given_x = T.nnet.softmax(T.dot(self.output_hidden_dropout, self.W2) + self.b2)

        
        #y_pred = T.argmax(p_y_given_x, axis=1)
        return -T.mean(T.log(p_y_given_x)[T.arange(y.shape[0]), y])
    
    
    def errors(self, y):

        return T.mean(T.neq(self.y_pred, y))

        