'''
Created on 10-May-2015

@author: amilgeorge
'''

import numpy as np
from scipy import linalg
from Visualizer import Visualizer
from DataLoader import DataLoader
from Util import check_create_observations_dir
import matplotlib.pyplot as plt
import os


class PCA(object):
    '''
    classdocs
    '''


    def __init__(self, params=None):
        '''
        Constructor
        '''
        self.num_components = 2
        
        self.W = None
        self.v = None
        
        
    
    
    def get_eig(self,X,num_components):
    
        v, W = linalg.eigh(X)
        
        ## Eigen vectors are column wise
        v, W = v[::-1], W[:, ::-1]
        
        v, W = v[:num_components],W[:,:num_components]
        
        return v, W
        
        
        
    def train(self,X):
        no_of_features = X.shape[1]
        N = X.shape[0];
        
        #Subtact Mean
        mean = X.mean(axis=0)
        X_std=X - mean;
        
        cov_X = np.dot(X_std.T,X_std) / (N-1)
        
        v, W = self.get_eig(cov_X,2)
        
       
        self.v = v
        self.W = W
        #return v, W
        
    def project(self, inpt):
        ##############
        #Subtact Mean#
        ##############
        #mean = inpt.mean(axis=0)
        #inpt=inpt - mean;    
        
        
        projected_data=np.dot(inpt,self.W)
            
        return projected_data
        
        #print cov_X;


def get_pairwise_plot(train_set_x,train_set_y):
    
    fig, plots = plt.subplots(10, 10)
    fig.set_size_inches(50, 50)
    plt.prism()
    pca = PCA()
    for i in xrange(10):
        for j in xrange(10):
            
    
            class_i = i;
            class_j = j;
            
            class1_indexes=[index for index,value in enumerate(train_set_y) if value==class_i]
            class2_indexes=[index for index,value in enumerate(train_set_y) if value==class_j]
            
            class1_data = train_set_x[class1_indexes,:]
            class2_data = train_set_x[class2_indexes,:]
            
            pca.train(np.vstack((class1_data,class2_data)));
            
            
            
            class1_proj=pca.project(class1_data)
            plots[i, j].plot(class1_proj[:,0],class1_proj[:,1],'o',markersize=3,color='red',alpha=0.5,label=class_i)
           
            if class_i!=class_j:        
                class2_proj=pca.project(class2_data)
                plots[i, j].plot(class2_proj[:,0],class2_proj[:,1],'o',markersize=3,color='green',alpha=0.5,label=class_j)
    

            plots[i, j].set_xticks(())
            plots[i, j].set_yticks(())
            
    return plt

def fit_mnist():
    print "Computing for MNIST"
    data_loader = DataLoader()
    train_set,valid_set,test_set=data_loader.load_data()        
    
    train_set_x,train_set_y=train_set  
    
    plt=get_pairwise_plot(train_set_x, train_set_y)
    
    obs_dir=check_create_observations_dir("PCA")
    target_path = os.path.join(obs_dir,"scatterplotMNIST.png")
    plt.savefig(target_path)
    print "THE END" 
    
def fit_cifar():
    print "Computing for CIFAR 10"
    data_loader = DataLoader()
    cifar_data=data_loader.load_cifar_data()        
    
    train_set_x=cifar_data['data']
    train_set_y=cifar_data['labels']  
    
    plt=get_pairwise_plot(train_set_x, train_set_y)
    
    obs_dir=check_create_observations_dir("PCA")
    target_path = os.path.join(obs_dir,"scatterplotCIFAR.png")
    plt.savefig(target_path)
    print "THE END" 

if __name__ == '__main__':
    fit_cifar()

            