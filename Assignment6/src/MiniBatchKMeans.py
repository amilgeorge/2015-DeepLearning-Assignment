'''
Created on 17-May-2015

@author: amilgeorge
'''
from DataLoader import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA
from Visualizer import display_bar
from Util import check_create_observations_dir


class MiniBatchKMeans(object):
    '''
    Clustering algorithm with mini batch support
    '''


    def __init__(self,num_clusters =100,max_iter=50, batch_size = 500,init = 'random'):
        '''
        Constructor
        '''
        self.max_iter = max_iter
        self.num_clusters=num_clusters
        self.batch_size = batch_size  
        if init == 'random':             
            self.cluster_init_method = self.init_centroids_from_sample
        else: 
            self.cluster_init_method = self.init_centroids
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

    '''
    D = numFeatues X NumCentroids
    Returns centroids of in matrix of the form NumCentroids X NumFeatures
    '''
    def init_centroids_from_sample(self,num_centroids,X):
        num_samples = X.shape[1]
        num_features = X.shape[0]
        
        #Randomly initialize centroids from a normal distribution and normalize them to unit length\
        
        center_indexes=np.random.randint(0,num_samples,size=self.num_clusters)
        D=X[:,center_indexes]
        return D
    
    def get_random_centroid(self):
        pass
    
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
        D, V =np.linalg.eig(cov)
        
        zca = (V.dot(np.diag((D+0.01)**-0.5).dot(V.T))).dot(normalized_patches.T)
        
        centers = self.cluster_init_method(self.num_clusters,zca)
        num_features = zca.shape[0]
        num_samples = zca.shape[1]
        
        n_batches = num_samples / self.batch_size
        
        for j in xrange(self.max_iter):
            print "Iteration ", j 
        
            sum_centroids = np.zeros((num_features,self.num_clusters))
            counts = np.zeros(self.num_clusters)
            loss = 0
            D=centers
            for batch_no in xrange(n_batches):
            
                zca_batch = zca[:,(batch_no * self.batch_size) : (batch_no + 1)*self.batch_size]
                num_samples_in_batch = zca_batch.shape[1]
                
                #NumCentroids X Samples
                temp = D.T.dot(zca_batch)
                #num_samples = temp.shape[1]
                    
                index_S = np.argmax(temp, axis=0)
                
                # S - numCentroids X numSamples
                S = np.zeros((temp.shape))
                for i in xrange(num_samples_in_batch):
                    S[index_S[i]][i]=1
                
                
                #X - numFeatures X numSamples 
                X = zca_batch
                
                #D = numFeatues X NumCentroids
                D = X.dot(S.T) 
                
                sum_centroids = sum_centroids + D
                counts = counts + np.sum(S,axis=1)
                
           
           
            for i in xrange(self.num_clusters):
                if counts[i] != 0: 
                    centers[:,i] = sum_centroids[:,i]/counts[i]  
                else:
                    # D[:,i] = D[:,i]*0  
                    pass
                   
            cluster_counts = counts           
                
                
                
            centers=self.__normalize_column(centers)
                
        return centers,cluster_counts
        
        
            
        
        
        
        
    def generate_patches(self,images,patch_size):
        '''
        images - [ image_no X Height X Width X Channels]
        '''
        print "Preparing patches" 
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
   
    #   data_row = data_row + data_row.min()
    data_row = data_row/data_row.max()
    img = data_row.reshape(patch_size[0],patch_size[1],3).astype('float32')
    #img = np.rollaxis(img,0,3)
    return img
    #plt.imshow(img)
    #plt.show()

def start(p = 12,num_clusters =100,max_iter=50, batch_size = 500,init = 'random' ):
    
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
    patch_size = [p,p]
#   
    kmeans = MiniBatchKMeans(num_clusters,max_iter,batch_size,init)  
    
    patches_img = kmeans.generate_patches(images, patch_size)
    
#     plt.imshow(patches_img[0,:,:,:])
#     plt.show();
    # Convert to matrix form rows X cols
    patches=patches_img.reshape(patches_img.shape[0],-1)
#     i=display(patches[0,:], patch_size)
#     plt.imshow(i)
#     plt.show()
    
    
    #pre-processing
    
    centers,counts = kmeans.fit(patches)
    
    fig = plt.figure()
    disp_row_size = np.ceil(np.sqrt(kmeans.num_clusters))
    
    for i in xrange(kmeans.num_clusters):
        subplot = fig.add_subplot(disp_row_size, disp_row_size, i+1)    
        subplot.get_xaxis().set_visible(False)
        subplot.get_yaxis().set_visible(False)
        img = display(centers[:,i], patch_size)
        subplot.imshow(img, interpolation='none')
    
    #plt.show()
    
    directory=check_create_observations_dir()
    
    plt.savefig(directory+'/repFields.png')
#     patch_test=patches[0,:,:,:]
#     plt.imshow(patch_test)
#     plt.show()
    
    display_bar(counts,directory+'/clusterCount.png')
    
    
    print "THE END" 
    
    
if __name__ == '__main__':
    start()