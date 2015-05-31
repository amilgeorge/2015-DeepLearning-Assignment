'''
Created on 22-Apr-2015

@author: amilgeorge
'''
import theano.tensor as T
from theano import function

x = T.dmatrix('x')
s = 1 / (1 + T.exp(-x))
logistic = function([x], s)
print logistic([[0, 1], [-1, -2]])
print logistic([0])