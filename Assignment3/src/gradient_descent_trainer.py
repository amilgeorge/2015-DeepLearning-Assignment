'''
Created on 06-May-2015

@author: amilgeorge
'''

from trainer import trainer
import numpy
import os
import sys
import time

import theano
import theano.tensor as T
from DataLoader import DataLoader
from Util import check_create_observations_dir, save_to_file
from Visualizer import display

class gradient_descent_trainer(trainer):
    '''
    classdocs
    '''


    def __init__(self, learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=1000, batch_size=20):
        '''
        Constructor
        learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=1000,
             dataset='mnist.pkl.gz', batch_size=20, n_hidden=500
        '''
        trainer.__init__(self)
        self.batch_size = batch_size
        self.L1_lambda = L1_reg;
        self.L2_lambda = L2_reg;
        self.learning_rate =learning_rate;
        self.n_epochs=n_epochs;
        self.early_stopping_threshold = 0.995
        
        
    def train_NN(self, nn):
        
        trainer.train_NN(self, nn) 
                  
        data_loader = DataLoader()
        datasets = data_loader.load_shared_data()
    
        train_set_x, train_set_y = datasets[0]
        valid_set_x, valid_set_y = datasets[1]
        test_set_x, test_set_y = datasets[2]
    
        # compute number of minibatches for training, validation and testing
        batch_size = self.batch_size;
        
        n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
        n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size
        
        index = T.lscalar()  # index to a [mini]batch
        x = nn.input  # the data is presented as rasterized images
        y = T.ivector('y') 
        
        cost = (nn.negative_log_likelihood_dropout(y)
        + self.L1_lambda * nn.L1
        + self.L2_lambda * nn.L2
        )
        
        train_err_model = theano.function(
        inputs=[index],
        outputs=nn.errors(y),
        givens={
            x: train_set_x[index * batch_size:(index + 1) * batch_size],
            y: train_set_y[index * batch_size:(index + 1) * batch_size]
        }
        )
        
        test_err_model = theano.function(
        inputs=[index],
        outputs=nn.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
        )

        validate_err_model = theano.function(
        inputs=[index],
        outputs=nn.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        })
                                         
        gparams = [T.grad(cost, param) for param in nn.params]
        
        updates = [
            (param, param - self.learning_rate * gparam)
            for param, gparam in zip(nn.params, gparams)
        ]
                                         
        train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
        )
       

        
        
        def validate():
        
            validation_losses = [validate_err_model(i)
                                         for i in xrange(n_valid_batches)]
            this_validation_loss = numpy.mean(validation_losses)
           
            return this_validation_loss;
        
        def test():
            test_losses = [test_err_model(i)
                                         for i in xrange(n_test_batches)]
            this_test_loss = numpy.mean(test_losses)
           
            return this_test_loss;    
        
        def train():
            train_losses = [train_err_model(i)
                                         for i in xrange(n_train_batches)]
            this_train_loss = numpy.mean(train_losses)
           
            return this_train_loss; 
        print '... training'
            #Train in mini batches
        minEpochs = 4
        validationFrequency =  n_train_batches;
        iteration = 0;
        bestValidationLoss = numpy.Inf;
        
        directory=check_create_observations_dir()
        self.output_directory = directory
        max_epoch_reached = False
        
        while not max_epoch_reached : 
            
            iteration = iteration + 1;
            epochNo = (iteration / n_train_batches) + 1
            batchId= iteration % n_train_batches;
            currentCost=train_model(batchId)
            
            #print "Cost = %f" %(currentCost)
            if iteration % validationFrequency == 0:
                validation_err = validate()
                train_err = train()
                test_err = test()
                
                self.add_train_data(epochNo, train_err, validation_err, test_err)
                print "Epoch no: %d Validation Loss = %f" %(epochNo,validation_err*100)
                if validation_err < bestValidationLoss:
                    bestValidationLoss = validation_err
                    
                if epochNo > minEpochs and validation_err *self.early_stopping_threshold > bestValidationLoss:
                    #print "------------------------Validation Loss = %f" %(validationLoss*100)
                    break;
             
        if epochNo >= self.n_epochs:
            max_epoch_reached = True
            
        testLoss=test()
        trainer.save_errors(self, directory)
        repfields_final_path=os.path.join(directory,"repFields.png")
        W_vals=nn.W1.get_value(borrow=True)
        
        display(W_vals,repfields_final_path)
        print  "iteration  %d complete. Cost = %f Best Validation Loss = %f Test Loss = %f" %(iteration,currentCost,bestValidationLoss *100,testLoss *100)   
