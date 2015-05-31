'''
Created on 26-May-2015

@author: amilgeorge
'''

from gradient_descent_trainer import gradient_descent_trainer

from NeuralNetwork import NeuralNetwork
from rms_prop_trainer import rms_prop_trainer

'''
Created on 25-May-2015

@author: amilgeorge
'''


import argparse
from argparse import ArgumentParser
from os.path import  dirname, join as path_join
import os
import textwrap

DEFAULT_NUM_HIDDEN_UNITS = 300

#GD Defaults
DEFAULT_GD_MAX_ITERATIONS = 1000
DEFAULT_GD_BATCH_SIZE = 600
DEFAULT_GD_LEARNING_RATE = 0.13
DEFAULT_GD_L1_LAMBDA = 0.00
DEFAULT_GD_L2_LAMBDA = 0.00

#Climin GD Defaults
DEFAULT_CLIMIN_GD_MAX_ITERATIONS = 1000
DEFAULT_CLIMIN_GD_BATCH_SIZE = 600
DEFAULT_CLIMIN_GD_LEARNING_RATE = 0.13
DEFAULT_CLIMIN_GD_L1_LAMBDA = 0.00
DEFAULT_CLIMIN_GD_L2_LAMBDA = 0.00

DEFAULT_RMSPROP_LEARNING_RATE=0.001
DEFAULT_RMSPROP_DECAY = 0.9
DEFAULT_RMSPROP_MOMENTUM = 0.0

GD_ALGO = "gd"
RMS_PROP_ALGO = "rmsprop"
CLIMIN_GD_ALGO = "climin_gd"

def _argparse():
    
    argparse1 = ArgumentParser('main_NN.py',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparse1.add_argument('-a', '--activation', help = 'activation to be used in the hidden layer 1. tanh 2. sigmoid  3. relu', default='tanh')
    argparse1.add_argument('-hu', type=int, help = 'number of neurons in the hidden layer', default=DEFAULT_NUM_HIDDEN_UNITS)
    argparse1.add_argument('-d', '--dropout_rate', help = 'Regularization Drop out rate ', default=0.0)
    subparsers = argparse1.add_subparsers(dest='algo')
    gd_parser = subparsers.add_parser(GD_ALGO,help = "Gradient Descent")
    rmsprop_parser = subparsers.add_parser(RMS_PROP_ALGO,help = "RMSProp algo")
    #climin_gd_parser = subparsers.add_parser(CLIMIN_GD_ALGO,help = "Climin Gradient Descent")
    
    
    ######################
    #GD Trainer Arguments#
    ######################
    gd_parser.add_argument('-t', type=int,help = 'maximum iterations', default=DEFAULT_GD_MAX_ITERATIONS)
    gd_parser.add_argument('-b', '--batch_size', help = 'size of batch',type=int, default=DEFAULT_GD_BATCH_SIZE)
    gd_parser.add_argument('-l', '--learning_rate', help = 'learning rate',type=float, default=DEFAULT_GD_LEARNING_RATE)
    gd_parser.add_argument('-l1', '--l1', help = 'L1 Regularization constant',type=float, default=DEFAULT_GD_L1_LAMBDA)
    gd_parser.add_argument('-l2', '--l2', help = 'L2 Regularization constant',type=float, default=DEFAULT_GD_L2_LAMBDA)
    
    #############################
    #CLIMIN GD Trainer Arguments#
    #############################
#     climin_gd_parser.add_argument('-t', type=int,help = 'maximum iterations', default=DEFAULT_CLIMIN_GD_MAX_ITERATIONS)
#     climin_gd_parser.add_argument('-b', '--batch_size', help = 'size of batch',type=int, default=DEFAULT_CLIMIN_GD_BATCH_SIZE)
#     climin_gd_parser.add_argument('-l', '--learning_rate', help = 'learning rate',type=float, default=DEFAULT_CLIMIN_GD_LEARNING_RATE)
#     climin_gd_parser.add_argument('-l1', '--l1', help = 'L1 Regularization constant',type=float, default=DEFAULT_CLIMIN_GD_L1_LAMBDA)
#     climin_gd_parser.add_argument('-l2', '--l2', help = 'L2 Regularization constant',type=float, default=DEFAULT_CLIMIN_GD_L2_LAMBDA)

    #############################
    #RMSProp Trainer Arguments#
    #############################
    rmsprop_parser.add_argument('-t', type=int,help = 'maximum iterations', default=DEFAULT_CLIMIN_GD_MAX_ITERATIONS)
    rmsprop_parser.add_argument('-b', '--batch_size', help = 'size of batch',type=int, default=DEFAULT_CLIMIN_GD_BATCH_SIZE)
    rmsprop_parser.add_argument('-l', '--learning_rate', help = 'learning rate',type=float, default=DEFAULT_RMSPROP_LEARNING_RATE)
    rmsprop_parser.add_argument('-l1', '--l1', help = 'L1 Regularization constant',type=float, default=DEFAULT_CLIMIN_GD_L1_LAMBDA)
    rmsprop_parser.add_argument('-l2', '--l2', help = 'L2 Regularization constant',type=float, default=DEFAULT_CLIMIN_GD_L2_LAMBDA)
    rmsprop_parser.add_argument('-d', '--decay', help = 'Decay parameter for the moving average.',type=float, default=DEFAULT_RMSPROP_DECAY)
    #rmsprop_parser.add_argument('-m', '--momentum', help = 'Momentum to use during optimization.',type=float, default=DEFAULT_RMSPROP_MOMENTUM)
    return argparse1




def main(args):
    #args_string = str(args)
   
    argp = _argparse().parse_args(args[1:])
    algo = argp.algo
#     help_f = open('help.md', 'w')
#     _argparse().print_help(file=help_f)
#     help_f.close()
    output_dir = None
    if algo==RMS_PROP_ALGO:
        classifier = NeuralNetwork(28 * 28,argp.hu,10,argp.activation,argp.dropout_rate);
        trainer=rms_prop_trainer(argp.learning_rate,argp.l1,argp.l2,argp.t,argp.batch_size,argp.decay)
        trainer.train_NN(classifier)
        output_dir=trainer.output_directory 

    else:
        classifier = NeuralNetwork(28 * 28,argp.hu,10,argp.activation,argp.dropout_rate);
        trainer=gradient_descent_trainer(argp.learning_rate,argp.l1,argp.l2,argp.t,argp.batch_size)
        trainer.train_NN(classifier)
        output_dir=trainer.output_directory 

    #print "output"
    #print output_dir
    cmd_file_path=os.path.join(output_dir,"command.txt")
    f = open(cmd_file_path,'w')
    f.write("python ")
    for a in args:
        f.write(str(a))
        f.write(" ")
    f.close()
    print "THE END"

if __name__ == '__main__':
    from sys import argv
    exit(main(argv))