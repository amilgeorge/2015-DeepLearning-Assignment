'''
Created on 17-May-2015

@author: amilgeorge
'''
from DataLoader import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA

class KMeans(object):
    '''
    classdocs
    '''


    def __init__(self,num_clusters = 64,max_iter=50):
        '''
        Constructor
        '''
        self.max_iter = max_iter
        self.num_clusters=num_clusters
    '''
    D = numFeatues X NumCentroids
    Returns centroids of in matrix of the form NumCentroids X NumFeatures
    '''
    
    def init_centroids(self,num_centroids,X):
        shape = X.shape
        num_features = X.shape[0]
        
        #Randomly initialize centroids from a normal distribution and normalize them to unit length
        mean=np.ones(num_features)
        cov = np.identity(num_features, 'float')
        x = np.random.multivariate_normal(mean,cov,num_centroids)
        norm_x = LA.norm(x, axis=1)
        
        D_T =x/norm_x[:,np.newaxis]
        return D_T.T
        #print "abc"
    
    def __normalize_column(self, matrix):
        norm = LA.norm(matrix,axis=0)
        normalized_x= matrix /norm[np.newaxis,:]
        return normalized_x
        
    def fit(self,X):
        #1. Normalize Inputs
        mean = X.mean(axis=0,keepdims = True)
        std = X.std(axis=0,keepdims = True)
        
        normalized_patches = (X - mean)/np.sqrt(std + 10)
        
        #2. whiten inputs
        
        cov = np.cov(normalized_patches.T)
        eig_val, V =np.linalg.eig(cov)
        
        zca = (V.dot(np.diag((eig_val+0.01)**-0.5).dot(V.T))).dot(normalized_patches.T)
        
        D = self.init_centroids(self.num_clusters,zca)
        
        for j in xrange(self.max_iter):
        
            #NumCentroids X Samples
            temp = D.T.dot(zca)
            num_samples = temp.shape[1]
                
            index_S = np.argmax(temp, axis=0)
            
            # S - numCentroids X numSamples
            S = np.zeros((temp.shape))
            for i in xrange(num_samples):
                S[index_S[i]][i]=1
            
            
            #X - numFeatures X numSamples 
            X = zca
            
            D = X.dot(S.T) 
            D=self.__normalize_column(D)
            
        return D
        
        
            
        
        
        
        
    def generate_patches(self,images,patch_size):
        '''
        images - [ image_no X Height X Width X Channels]
        '''
        
        num_images = images.shape[0]
        #num_images=1000
        image_rows = images.shape[1];
        image_cols = images.shape[2];
        patch_size_rows = patch_size[0]
        patch_size_cols = patch_size[1]
        patches = np.zeros((0,patch_size_rows,patch_size_cols,3))
        for i in xrange(num_images):
            img = images[i,:,:,:]
            r= np.random.randint(0,image_rows - patch_size_rows + 1)
            c= np.random.randint(0,image_cols - patch_size_cols + 1)
            p=img[r:r+patch_size_rows,c:c+patch_size_cols,:]
#             plt.imshow(p)
#             plt.show()
            p=p.reshape(1,patch_size_rows,patch_size_cols,3)
#             plt.imshow(p[0])
#             plt.show()
#             plt.imshow(img)
#             plt.show()
            patches = np.concatenate((patches,p),axis=0)
        
        
        return patches

def display(data_row,patch_size):
    
    data_row = data_row - data_row.min()
    data_row = data_row/data_row.max()
    img = data_row.reshape(3,patch_size[0],patch_size[1]).astype('float32')
    img = np.rollaxis(img,0,3)
    return img
    #plt.imshow(img)
    #plt.show()
    
if __name__ == '__main__':
    data_loader = DataLoader()
    cifar_data = data_loader.load_cifar_data()     
    images = cifar_data['data'].reshape((-1,3,32,32)).astype('float32')
#     img_test = images[2,:,:,:]
#     img_test = np.rollaxis(img_test,0,3)
#     img_test = img_test[:,:,::-1]
#     plt.imshow(img_test)
#     plt.show()

    images = np.rollaxis(images,1,4)
    images = images[:,:,:,::-1]
    
    num_patches = images.shape[0]
    patch_size = [12,12]
#   
    kmeans = KMeans()  
    
    patches = kmeans.generate_patches(images, patch_size)
    
    # Convert to matrix form rows X cols
    patches=patches.reshape(patches.shape[0],-1)
    
    #pre-processing
    
    centers = kmeans.fit(patches)
    display(centers[:,0],patch_size)
    
    fig = plt.figure()
    disp_row_size = np.ceil(np.sqrt(kmeans.num_clusters))
    
    for i in xrange(kmeans.num_clusters):
        subplot = fig.add_subplot(disp_row_size, disp_row_size, i)    
        subplot.get_xaxis().set_visible(False)
        subplot.get_yaxis().set_visible(False)
        img = display(centers[:,i], patch_size)
        subplot.imshow(img, interpolation='none')
    
    plt.show()
#     patch_test=patches[0,:,:,:]
#     plt.imshow(patch_test)
#     plt.show()
    
    
    
    print "THE END" 