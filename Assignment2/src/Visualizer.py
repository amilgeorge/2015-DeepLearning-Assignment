'''
Created on 26-May-2015

@author: amilgeorge
'''
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os


def display(W,filename):
    cols = W.shape[1]
    fig= plt.figure()
    for i in xrange(cols):
        data_row = W[:,i]
        img=dispaly_row(data_row)
        subplot = fig.add_subplot(2, 5, i+1)    
        subplot.get_xaxis().set_visible(False)
        subplot.get_yaxis().set_visible(False)
        subplot.imshow(img, cmap = cm.gray)  # @UndefinedVariable
    plt.savefig(filename)
    
def dispaly_row(data_row):
  
    data_row = data_row - data_row.min()
   
    #   data_row = data_row + data_row.min()
    data_row = data_row/data_row.max()
    img = data_row.reshape(28,28).astype('float32')
    return img

def plot_errors(file_dir):
    validation_err_file = os.path.join(file_dir,'validation_err.txt')
    train_err_file = os.path.join(file_dir,'train_err.txt')
    test_err_file = os.path.join(file_dir,'test_err.txt')
    target_file = os.path.join(file_dir,'error.png')
    train_errors=np.loadtxt(train_err_file, delimiter=" ")
    valid_errors=np.loadtxt(validation_err_file, delimiter=" ")
    test_errors=np.loadtxt(test_err_file, delimiter=" ")
    train_errors [:,1]= train_errors[:,1] *100
    test_errors[:,1] = test_errors[:,1] *100
    valid_errors[:,1] = valid_errors[:,1] *100
    fig= plt.figure()
    plt.plot(train_errors[:,0],train_errors[:,1],'r--',label="Train")
    plt.plot(valid_errors[:,0],valid_errors[:,1],'b--',label = "Validation")
    plt.plot(test_errors[:,0],test_errors[:,1],'g--',label="Test")
    plt.xlabel('Iteration Number')
    plt.ylabel('Error')
    plt.title('Error Curves')
    plt.legend()
    plt.savefig(target_file)
    plt.close()
    
    

if __name__ == '__main__':
    plot_errors("../Observations/2015-05-27 20:43:42")