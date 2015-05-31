'''
Created on 26-May-2015

@author: amilgeorge
'''

from PCA import fit_mnist, fit_cifar

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
    
    argparse1 = ArgumentParser('main_LR.py',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparse1.add_argument('-d', help = "data set to be used 1. \'mnist\', \n 2. \'cifar\'",default="mnist")
    

    return argparse1




def main(args):
    #args_string = str(args)
   
    argp = _argparse().parse_args(args[1:])
    dataset = argp.d
    help_f = open('help.md', 'w')
    _argparse().print_help(file=help_f)
    help_f.close()
   
    if dataset=="mnist":
        fit_mnist()
    elif dataset == "cifar":
        fit_cifar()
    else:
        fit_mnist()
    
    print "THE END"

if __name__ == '__main__':
    from sys import argv
    exit(main(argv))