'''
Created on 22-Apr-2015

@author: amilgeorge
'''

import theano.tensor as T
from theano import function
x = T.dscalar('x')
y = T.dscalar('y')
z = x + y
f = function([x, y], z)

print f(2,3)