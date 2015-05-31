'''
Created on 06-May-2015

@author: amilgeorge
'''


import cPickle
import gzip
import os


import numpy as np

import theano
import theano.tensor as T
from Util import save_to_file
from Visualizer import plot_errors

class trainer(object):
    '''
    classdocs
    '''
      


    def __init__(self):
        '''
        Constructor
        '''
        self.validation_errors = []
        self.test_errors = []
        self.train_errors = []
        self.iteration_nos = []
        self.output_directory=None
  
    def train_NN(self,nn): 
        pass   
    
    def add_train_data(self,iteration_no,train_err,validation_err,test_err):
        self.iteration_nos = np.append(self.iteration_nos,iteration_no)
        self.validation_errors = np.append(self.validation_errors,validation_err)
        self.train_errors = np.append(self.train_errors,train_err)
        self.test_errors = np.append(self.test_errors,test_err)
        
    
    def save_errors(self,directory):
        validation_err_file = os.path.join(directory,'validation_err.txt')
        train_err_file = os.path.join(directory,'train_err.txt')
        test_err_file = os.path.join(directory,'test_err.txt')
        
        save_to_file(np.c_[(self.iteration_nos,self.validation_errors)],validation_err_file)
        save_to_file(np.c_[(self.iteration_nos,self.train_errors)],train_err_file)
        save_to_file(np.c_[(self.iteration_nos,self.test_errors)],test_err_file)
        plot_errors(directory)
    
