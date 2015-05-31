'''
Created on 27-May-2015

@author: amilgeorge
'''
from trainer import trainer
from DataLoader import DataLoader
import climin
import numpy as np
import theano
import theano.tensor as T
import climin.initialize
import climin.util
from Util import  check_create_observations_dir
from Visualizer import display
import os

class climin_trainer(trainer):
    '''
    classdocs
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
        
        params=np.empty((28*28)*10+10);
    
        climin.initialize.randomize_normal(params,0,1) 
        params = params/(28)
    
        lr.setParams(params);
        
        x=lr.x
        y=lr.y
        cost = (
                lr.negative_log_likelihood(y)
                + self.L1_lambda * lr.L1
                + self.L2_lambda * lr.L2_sqr
            )
                 
        g_W = T.grad(cost=cost, wrt=lr.W)
        g_b = T.grad(cost=cost, wrt=lr.b)   
                
        g_W_model = theano.function(
                                    inputs=[x,y],
                                    outputs=g_W         
                                    )
        g_b_model = theano.function(
                                    inputs=[x,y],
                                    outputs=g_b         
                                    )    
        
        
        batch_size = self.batch_size
        index = T.lscalar()
        
        test_err_model = theano.function(
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
        
        validate_err_model = theano.function(
        inputs=[index],
        outputs=lr.zeroOneLoss(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        })

                # compute number of minibatches for training, validation and testing
        batch_size = self.batch_size;
        
        n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
        n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size
        
        def train_error():
            train_losses = [train_err_model(i)
                                         for i in xrange(n_train_batches)]
            this_train_losses = np.mean(train_losses)
           
            return this_train_losses;
        
        def validate_error():
        
            validation_losses = [validate_err_model(i)
                                         for i in xrange(n_valid_batches)]
            this_validation_loss = np.mean(validation_losses)
           
            return this_validation_loss;
        
        def test_error():
            test_losses = [test_err_model(i)
                                         for i in xrange(n_test_batches)]
            this_test_loss = np.mean(test_losses)
           
            return this_test_loss;           
                
        def d_loss_wrt_pars(parameters, inpt, targets):
                lr.setParams(parameters)
               
               
                gwValue = g_W_model(inpt,targets)
                gbValue = g_b_model(inpt,targets)
               
                return np.concatenate([gwValue.flatten(),gbValue])   
        
        args = ((i, {}) for i in climin.util.iter_minibatches([train_set_x.eval(), train_set_y.eval()], self.batch_size, [0, 0]))
   
        opt = climin.GradientDescent(params, d_loss_wrt_pars, step_rate=self.learning_rate, momentum=0.0, args=args)
        
        validation_frequency = n_train_batches
        directory=check_create_observations_dir()  
        self.output_directory = directory
        bestValidationLoss = np.Inf;
        for info in opt:
            if info['n_iter'] % validation_frequency ==0:
                epoch_no = info['n_iter']/n_train_batches
                
                train_err=train_error()
                validation_err = validate_error()
                test_err = test_error()
                self.add_train_data(epoch_no, train_err, validation_err, test_err)
                if epoch_no % 10 ==0:
                    repfields_path=os.path.join(directory,"repFields"+str(epoch_no).zfill(3)+'.png')
                    W_vals=lr.W.get_value(borrow=True)
                    display(W_vals,repfields_path)
                
                if epoch_no >= self.n_epochs:
                    break
                
                if validation_err < bestValidationLoss:
                    bestValidationLoss = validation_err
    #                     
                if  validation_err *0.99 > bestValidationLoss:
                      
                        print "Best Validation Error : %f Validation err:%f  " %(bestValidationLoss,validation_err)
                        break;
                
#                 if epoch_no > 15 and train_err > 0.9* validation_err:
#                     break
                
                print "Iteration no: %d Validation error = %f" %(epoch_no,validation_err*100)
            
         
        trainer.save_errors(self, directory)
        repfields_final_path=os.path.join(directory,"repFields.png")
        W_vals=lr.W.get_value(borrow=True)
        display(W_vals,repfields_final_path)