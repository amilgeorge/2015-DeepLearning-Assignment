'''
Created on 11-May-2015

@author: amilgeorge
'''

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm


def display_reconstructions(inp,out,filename):
    
    
    num_images =inp.shape[0]
    fig, plots = plt.subplots(num_images, 2)
    fig.suptitle('Input Vs Reconstructed Input', fontsize=14, fontweight='bold')
    for i in xrange(num_images):
        data_row1 = inp[i,:] 
        data_row2 = out[i,:] 
        img1=dispaly_row(data_row1)
        img2=dispaly_row(data_row2)
         
        plots[i,0].get_xaxis().set_visible(False)
        plots[i,0].get_yaxis().set_visible(False)
       
        plots[i,0].imshow(img1) 
     
        plots[i,1].get_xaxis().set_visible(False)
        plots[i,1].get_yaxis().set_visible(False)
         
        plots[i,1].imshow(img2)
         
   
       
    plt.savefig(filename) 
       
def display_sparse_encoder(W,filename):
    W_cols = W.shape[1]
    fig= plt.figure()
    fig.set_size_inches(100, 100)
    disp_row_size = np.ceil(np.sqrt(W_cols))
    subplot_rows =disp_row_size
    subplot_cols =disp_row_size
    for i in xrange(W_cols):
        data_row = W[:,i]
        img=dispaly_row(data_row)
        subplot = fig.add_subplot(subplot_rows, subplot_cols, i+1)    
        subplot.get_xaxis().set_visible(False)
        subplot.get_yaxis().set_visible(False)
        subplot.imshow(img)  # @UndefinedVariable
    
    plt.set_cmap('spectral')
    plt.savefig(filename)
   
    
def dispaly_row(data_row):
  
    data_row = data_row - data_row.min()
   
    #   data_row = data_row + data_row.min()
    data_row = data_row/data_row.max()
    img = data_row.reshape(28,28).astype('float32')
    return img

class Visualizer(object):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
    def scatter(self,data):
        color=['red','blue','green']
        i=0;
        for d in data:
            class_name=d[0]
            class_data=d[1]
            plt.plot(class_data[:,0],class_data[:,1],'o',markersize=7,color=color[i],alpha=0.5,label=class_name)
            max_val=np.amax(class_data, axis=0)
            min_val=np.amin(class_data, axis=0)
            x_lim = [min_val[0]-10,max_val[0]+10]
            y_lim = [min_val[1]-10,max_val[1]+10]
            #pyplot.xlim(x_lim)
            #pyplot.ylim(y_lim)
            i = i+1
        plt.legend()
        plt.show() 
            

if __name__ == '__main__':
    
    test = np.matrix([[1,2],[2,3],[3,6]])
    plt.plot(test[:,0],test[:,1],'o',markersize=7,color='blue',alpha=0.5,label='class1')
   
    plt.xlim()
    plt.ylim([-10,10])
    plt.legend()
    plt.show()   