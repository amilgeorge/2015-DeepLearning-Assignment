'''
Created on 21-May-2015

@author: amilgeorge
'''
import numpy as np
from KMeans import KMeans

if __name__ == '__main__':
    mean = [0,0,0]
    cov = [[1,1,1],[0,1,0]]
    
    import matplotlib.pyplot as plt
    #x = np.random.multivariate_normal(mean,cov,5000)
    k=KMeans()
    s = np.ones((2,3))
    k.init_centroids(2, s)
    #plt.plot(x,y,'x'); plt.axis('equal'); plt.show()
    print "Theheheh"