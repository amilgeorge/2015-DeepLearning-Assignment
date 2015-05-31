'''
Created on 25-May-2015

@author: amilgeorge
'''


import argparse
from argparse import ArgumentParser
from os.path import  dirname, join as path_join
from MiniBatchKMeans import start
import textwrap

BH_TSNE_BIN_PATH = path_join(dirname(__file__), 'bh_tsne')

# Default hyper-parameter values from van der Maaten (2013)
DEFAULT_NO_CLUSTERS = 64
DEFAULT_CLUSTER_CENTER_INIT_METHOD = 'random'
MAX_ITERATIONS = 50
BATCH_SIZE = 500
DEFAULT_PATCH_SIZE = 12

def _argparse():
    
    argparse1 = ArgumentParser('main_Kmeans.py',formatter_class=argparse.RawDescriptionHelpFormatter,description=textwrap.dedent('''\
...         Mini Batch K-Means
...         --------------------------------
            Mini Batch K means clustering algorithm for cifar data set
            --------------------------------
            --------------------------------
            Output
            --------------------------------
            Saves the following output files:
            1. repFields.png         - Visualization of cluster centers determined through KMeans
            2. clusterCount.png      - The number of samples in each cluster  
            --------------------------------
            --------------------------------
...         '''))
    argparse1.add_argument('-k', type=int, help = 'number of clusters',
                          default=DEFAULT_NO_CLUSTERS)
    argparse1.add_argument('-c', help = "cluster center initialization method 1. \'random\', \n 2. \'normal\'",
            default=DEFAULT_CLUSTER_CENTER_INIT_METHOD)
    # 0.0 for theta is equivalent to vanilla t-SNE
    argparse1.add_argument('-t', type=int,help = 'maximum iterations', default=MAX_ITERATIONS)
    argparse1.add_argument('-b', '--batch_size', help = 'size of batch',type=int, default=BATCH_SIZE)
    
  
    argparse1.add_argument('-p', '--patch_size',help = 'size of the patch that should be used. ', type=int,default=DEFAULT_PATCH_SIZE )

    return argparse1




def main(args):
   
    argp = _argparse().parse_args(args[1:])
    #f = open('main_Kmeans.md', 'w')
    #_argparse().print_help(file=f)
    #,num_clusters =100,max_iter=50, batch_size = 500,init = 'random' 
    start(argp.patch_size,argp.k,argp.t,argp.batch_size,argp.c)
    
    print "THE END"

if __name__ == '__main__':
    from sys import argv
    exit(main(argv))