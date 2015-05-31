'''
Created on 26-May-2015

@author: amilgeorge
'''


from SparseAutoEncoder import SparseAutoEncoder
from sgd_trainer import sgd_trainer
from Visualizer import  display_sparse_encoder,display_reconstructions
from Util import check_create_observations_dir
from DataLoader import DataLoader

'''
Created on 25-May-2015

@author: amilgeorge
'''


import argparse
from argparse import ArgumentParser
from os.path import  dirname, join as path_join
import os
import textwrap


#GD Defaults
DEFAULT_NUM_HIDDEN_UNITS = 500
DEFAULT_GD_MAX_ITERATIONS = 20
DEFAULT_GD_BATCH_SIZE = 600
DEFAULT_GD_LEARNING_RATE = 0.01
DEFAULT_GD_SPARSITY_LAMBDA = 0.01
DEFAULT_GD_L2_LAMBDA = 0.00

#Climin GD Defaults

DEFAULT_CLIMIN_GD_MAX_ITERATIONS = 1000
DEFAULT_CLIMIN_GD_BATCH_SIZE = 600
DEFAULT_CLIMIN_GD_LEARNING_RATE = 0.13
DEFAULT_CLIMIN_GD_SPARSITY_LAMBDA = 0.01
DEFAULT_CLIMIN_GD_L2_LAMBDA = 0.00

DEFAULT_RMSPROP_LEARNING_RATE=0.001
DEFAULT_RMSPROP_DECAY = 0.9
DEFAULT_RMSPROP_MOMENTUM = 0.0

GD_ALGO = "gd"
RMS_PROP_ALGO = "rmsprop"
CLIMIN_GD_ALGO = "climin_gd"

def _argparse():
    
    argparse1 = ArgumentParser('main_sparse_autoencoder.py',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
   
    argparse1.add_argument('-hu', type=int,help = 'number of hidden units', default=DEFAULT_NUM_HIDDEN_UNITS)
    argparse1.add_argument('-t', type=int,help = 'maximum iterations', default=DEFAULT_GD_MAX_ITERATIONS)
    argparse1.add_argument('-b', '--batch_size', help = 'size of batch',type=int, default=DEFAULT_GD_BATCH_SIZE)
    argparse1.add_argument('-l', '--learning_rate', help = 'learning rate',type=float, default=DEFAULT_GD_LEARNING_RATE)
    argparse1.add_argument('-r', help = 'reconstruction cost function 1. \'cross_entropy\' 2. \'sqr\' - squared error function ', default='cross_entropy')
    argparse1.add_argument('-sc', help = 'sparsity cost function 1. \'kl\' - KL Divergence 2. \'l1\' - L1 penalty ', default='kl')

    argparse1.add_argument('-sl', '--sl', help = 'sparsity regularization constant',type=float, default=DEFAULT_GD_SPARSITY_LAMBDA)
   
    return argparse1




def main(args):
    #args_string = str(args)
   
    argp = _argparse().parse_args(args[1:])
    
    sae = SparseAutoEncoder(28*28,argp.hu)
    data_loader=DataLoader()
    
    datasets = data_loader.load_data()
    
   
    
    trainer = sgd_trainer(argp.batch_size,argp.learning_rate,argp.sl,argp.t,argp.r,argp.sc)
    trainer.trainAutoEncoder(sae);
    W=sae.W1.get_value(borrow=True)
    
    out_dir = check_create_observations_dir("AutoEncoder")
    target_file = os.path.join(out_dir,"autoencoderfilter.png")
    display_sparse_encoder(W,target_file)
    
    test_set_x,test_set_y = datasets[2]
    test_inpt = test_set_x[:10,:]
    
    mnist_vis_file = os.path.join(out_dir,"autoencoderrec.png")
    display_reconstructions(test_inpt,sae.encode(test_inpt),mnist_vis_file)
    
    cmd_file_path=os.path.join(out_dir,"command.txt")
    f = open(cmd_file_path,'w')
    f.write("python ")
    for a in args:
        f.write(str(a))
        f.write(" ")
    f.close()
    print "THE END"
    
    
    print "THE END"

if __name__ == '__main__':
    from sys import argv
    exit(main(argv))