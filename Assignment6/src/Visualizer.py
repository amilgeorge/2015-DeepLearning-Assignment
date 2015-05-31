'''
Created on 24-May-2015

@author: amilgeorge
'''
import numpy as np
import matplotlib.pyplot as plt

def display_bar(array,savefile_path):
    
    array_len = array.shape[0]
    plt.figure(figsize=(30,10))
    #fig, ax = plt.subplots()  
    plt.subplot(1,1,1)
    width = 0.5   
    
    index=np.arange(array_len)
    plt.bar(index,array,width)
    plt.xlabel('Cluster No')
    plt.ylabel('Size')
    x_axis = np.arange(array_len);
    plt.xticks(x_axis+width/2 , x_axis+1)
    plt.savefig(savefile_path)

if __name__ == '__main__':
    counts = np.zeros(100)
    counts[0]=2
    counts[2]=1
    display_bar(counts,"test1.png")