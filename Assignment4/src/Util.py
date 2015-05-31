'''
Created on 25-May-2015

@author: amilgeorge
'''

import time
import datetime
import os;
import numpy as np
 
def check_create_observations_dir(project = None): 
    ts = time.time()
   
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
   
    if project is None:
        directory =  os.path.join("..","Observations",st)
    else:
        directory = os.path.join("..","Observations",project,st)
        
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    return directory



def save_to_file(numpy_array, fqn ):
    np.savetxt(fqn, numpy_array, fmt='%.18g', delimiter=' ',header="No\tError", newline=os.linesep)
    
if __name__ == '__main__':
    fname = "../Observations/test.txt"
    x=[0,2,5,6]
    y=[3,643,453,5]
    save_to_file(np.c_[(x,y)], fname)
    a=np.loadtxt(fname, delimiter=" ")
    print a