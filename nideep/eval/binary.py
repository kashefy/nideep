'''
Created on Apr 21, 2017

@author: kashefy
'''
import numpy as np
from sklearn import metrics

class Binary(object):
    '''
    classdocs
    '''
    def set_data(self, gt, preds):
        
        self.num_classes = gt.shape[-1]
        if self.num_classes != len(self.classnames):
            raise ValueError("classnames (%s) do not much no. of classes (%d)" % ','.join(self.classnames),
                             self.num_classes)

    def __init__(self, classnames):
        '''
        Constructor
        '''
        self.classnames = classnames
        