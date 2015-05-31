'''
Created on 06-May-2015

@author: amilgeorge
'''

from trainer import trainer
import numpy


import theano
import theano.tensor as T
from DataLoader import DataLoader
from Util import check_create_observations_dir, save_to_file
from Visualizer import display
import os

import numpy as np

class gradient_descent_trainer(trainer):
    '''
    Mini Batch GD Optimizer for Logistic 
    '''


    def __init__(self, batch_size=600,learning_rate=0.13, L1_lambda=0.00, L2_lambda=0.0000, n_epochs=1000):
        '''
        
        '''
        trainer.__init__(self)
        self.batch_size = batch_size
        self.L1_lambda = L1_lambda;
        self.L2_lambda = L2_lambda;
        self.learning_rate =learning_rate;
        self.n_epochs=n_epochs;
        
        
    def train_LR(self, lr):
        
        trainer.train_LR(self, lr) 
                  
        dataloader = DataLoader()
        datasets =dataloader.load_shared_data()
    
        train_set_x, train_set_y = datasets[0]
        valid_set_x, valid_set_y = datasets[1]
        test_set_x, test_set_y = datasets[2]
    
        # compute number of minibatches for training, validation and testing
        batch_size = self.batch_size;
        
        n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
        n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size
        
        index = T.lscalar()  # index to a [mini]batch
        x = lr.x  # the data is presented as rasterized images
        y = lr.y 
        
        cost = (
        lr.negative_log_likelihood(y)
        + self.L1_lambda * lr.L1
        + self.L2_lambda * lr.L2_sqr
        )
        
        test_model = theano.function(
        inputs=[index],
        outputs=lr.zeroOneLoss(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
        )

        train_err_model = theano.function(
        inputs=[index],
        outputs=lr.zeroOneLoss(y),
        givens={
            x: train_set_x[index * batch_size:(index + 1) * batch_size],
            y: train_set_y[index * batch_size:(index + 1) * batch_size]
        }
        )
        
        validate_model = theano.function(
        inputs=[index],
        outputs=lr.zeroOneLoss(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        })
                                         
        g_W = T.grad(cost=cost, wrt=lr.W)
        g_b = T.grad(cost=cost, wrt=lr.b)


        updates = [(lr.W, lr.W - self.learning_rate * g_W),
               (lr.b, lr.b - self.learning_rate * g_b)]
                                         
        train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
        )

        
        def train_error():
            train_losses = [train_err_model(i)
                                         for i in xrange(n_train_batches)]
            this_train_losses = numpy.mean(train_losses)
           
            return this_train_losses;
        
        def validate_error():
        
            validation_losses = [validate_model(i)
                                         for i in xrange(n_valid_batches)]
            this_validation_loss = numpy.mean(validation_losses)
           
            return this_validation_loss;
        
        def test_error():
            test_losses = [test_model(i)
                                         for i in xrange(n_test_batches)]
            this_test_loss = numpy.mean(test_losses)
           
            return this_test_loss;   
         
        print '... training'
        #Train in mini batches
        minEpochs = 4
        validationFrequency = n_train_batches;
        iteration = 0;
        bestValidationLoss = numpy.Inf;
        max_epoch_reached = False
        
        
        
        directory=check_create_observations_dir()
        self.output_directory = directory
        while not max_epoch_reached :
            
            iteration = iteration + 1;
            epochNo = (iteration / n_train_batches) 
            batchId= iteration % n_train_batches;
            currentCost=train_model(batchId)
            
            if iteration % validationFrequency == 0:
                validation_err = validate_error()
                test_err = test_error()
                train_err = train_error()
                print "Epoch no: %d Validation Loss = %f" %(epochNo,validation_err*100)
                
                self.add_train_data(epochNo, train_err, validation_err, test_err)
                if epochNo %5 ==0:
                    W_vals=lr.W.get_value(borrow=True)
                    repfields_path=os.path.join(directory,"repFields"+str(epochNo).zfill(3)+'.png')
                    display(W_vals,repfields_path)
                
                if validation_err < bestValidationLoss:
                    bestValidationLoss = validation_err
                    
                if epochNo > minEpochs and validation_err *0.995 > bestValidationLoss:
                    #print "------------------------Validation Loss = %f" %(validationLoss*100)
                    break;
             
            if epochNo >= self.n_epochs:
                max_epoch_reached = True
            
        testLoss=test_error()
    
        print  "iteration  %d complete. Cost = %f Best Validation Loss = %f Test Loss = %f" %(iteration,currentCost,bestValidationLoss *100,testLoss *100)   
       
        
        trainer.save_errors(self, directory)
        repfields_final_path=os.path.join(directory,"repFields.png")
        W_vals=lr.W.get_value(borrow=True)
        display(W_vals,repfields_final_path)