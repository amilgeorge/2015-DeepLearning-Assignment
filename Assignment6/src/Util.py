'''
Created on 25-May-2015

@author: amilgeorge
'''

import time
import datetime
import os;
 
def check_create_observations_dir(project = None): 
    ts = time.time()
   
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
   
    if project is None:
        directory = ("../Observations/"+st)
    else:
        directory = ("../Observations/"+project+"/"+st)
        
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    return directory