'''
Created on 28-Apr-2015

@author: amilgeorge
'''

import cPickle
import gzip
import os


import numpy as np
from Util import check_create_observations_dir

import theano
import theano.tensor as T
from Visualizer import display

class LogisticRegression(object):
    '''
    Class implementing logistic regression
    Warning : Only tested on mnist data set
    
    '''


    def __init__(self, numIn, numOut):
     
        
        self.numIn = numIn
        self.numOut = numOut
            
        self.W = theano.shared(
            value=np.zeros(
                (numIn,numOut),
                dtype=theano.config.floatX  # @UndefinedVariable
            ),
            name='W',
            borrow=True
        )
        
        self.b = theano.shared(
            value=np.zeros(
                (numOut,),
                dtype=theano.config.floatX  # @UndefinedVariable
            ),
            name='b',
            borrow=True
        )
        
       
        self.x = T.matrix('x') ;
        self.y = T.ivector('y') ;
        
        self.p_y_given_x = T.nnet.softmax(T.dot(self.x,self.W) + self.b)

        
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        
        ######################## Regularization #########################
        self.L1 = (
            abs(self.W).sum()
        )

       
        self.L2_sqr = (
            (self.W ** 2).sum()
        )
        ###################################################################
        
        
        
    def zeroOneLoss(self,y):   
        return T.mean(T.neq(self.y_pred, y)) 
        
    def negative_log_likelihood(self, y):
    
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
   
    '''
    Convinience methods for supporting climin optimizers
    '''      
    def setParams(self,params):
        w,b=self.unpack_parameters(params)
        self.W.set_value(w);
        self.b.set_value(b);
        pass
    
    
     
    def d_loss_wrt_pars(self,parameters, inpt, targets):
        self.setParams(parameters)
        x=self.x
        y=self.y
        cost=self.negative_log_likelihood(y); 
        g_W = T.grad(cost=cost, wrt=self.W)
        g_b = T.grad(cost=cost, wrt=self.b)   
        
        gwValue = self.g_W_model(inpt,targets)
        gbValue = self.g_b_model(inpt,targets)
        #print gwValue
        return np.concatenate([gwValue.flatten(),gbValue])
    
    def getPackedParameters(self):
        w=self.W.get_value(borrow = True);
        b =self.b.get_value(borrow = True);
        wmat=w.reshape(1,self.numIn*self.numOut);
        warr=np.asarray(wmat)[0];
            
        bmat=b.reshape(1,self.numOut);
        barr=np.asarray(bmat)[0];
            
        packedParams=np.concatenate(warr,barr)
        return packedParams   
    
    def unpack_parameters(self,pars):
        w = pars[:self.numIn * self.numOut].reshape((self.numIn, self.numOut))
        b = pars[self.numIn * self.numOut:].reshape((1, self.numOut))
        #print b,b[0]
        return w, b[0]



    
