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
from Util import  save_to_file
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
        
    def train_LR(self,lr): 
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
    
   # def load_data(self,dataset='mnist.pkl.gz'):


    #############
    # LOAD DATA #
    #############

    # Download the MNIST dataset if it is not present
#         data_dir, data_file = os.path.split(dataset)
#         if data_dir == "" and not os.path.isfile(dataset):
#             # Check if dataset is in the data directory.
#             new_path = os.path.join(
#                 os.path.split(__file__)[0],
#                 "..",
#                 "data",
#                 dataset
#             )
#             if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
#                 dataset = new_path
#     
#         if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
#             import urllib
#             origin = (
#                 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
#             )
#             print 'Downloading data from %s' % origin
#             urllib.urlretrieve(origin, dataset)
#     
#         print '... loading data'
#     
#         # Load the dataset
#         f = gzip.open(dataset, 'rb')
#         train_set, valid_set, test_set = cPickle.load(f)
#         f.close()
#         #train_set, valid_set, test_set format: tuple(input, target)
#         #input is an numpy.ndarray of 2 dimensions (a matrix)
#         #witch row's correspond to an example. target is a
#         #numpy.ndarray of 1 dimensions (vector)) that have the same length as
#         #the number of rows in the input. It should give the target
#         #target to the example with the same index in the input.
#     
#         def shared_dataset(data_xy, borrow=True):
#             """ Function that loads the dataset into shared variables
#     
#             The reason we store our dataset in shared variables is to allow
#             Theano to copy it into the GPU memory (when code is run on GPU).
#             Since copying data into the GPU is slow, copying a minibatch everytime
#             is needed (the default behaviour if the data is not in a shared
#             variable) would lead to a large decrease in performance.
#             """
#             data_x, data_y = data_xy
#             shared_x = theano.shared(numpy.asarray(data_x,
#                                                    dtype=theano.config.floatX),  # @UndefinedVariable
#                                      borrow=borrow)
#             shared_y = theano.shared(numpy.asarray(data_y,
#                                                    dtype=theano.config.floatX),  # @UndefinedVariable
#                                      borrow=borrow)
#             # When storing data on the GPU it has to be stored as floats
#             # therefore we will store the labels as ``floatX`` as well
#             # (``shared_y`` does exactly that). But during our computations
#             # we need them as ints (we use labels as index, and if they are
#             # floats it doesn't make sense) therefore instead of returning
#             # ``shared_y`` we will have to cast it to int. This little hack
#             # lets ous get around this issue
#             return shared_x, T.cast(shared_y, 'int32')
#     
#         test_set_x, test_set_y = shared_dataset(test_set)
#         valid_set_x, valid_set_y = shared_dataset(valid_set)
#         train_set_x, train_set_y = shared_dataset(train_set)
#     
#         rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
#                 (test_set_x, test_set_y)]
#         return rval