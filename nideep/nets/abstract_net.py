'''
Created on Jul 14, 2017

@author: kashefy
'''
from __future__ import division, print_function, absolute_import
from abc import ABCMeta, abstractmethod

class AbstractNet(object):
    '''
    classdocs
    '''
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def _init_learning_params(self):
        pass  
    
    @abstractmethod
    def build(self):
        pass
    
    def _config_scopes(self):
        self.var_scope = self.prefix
        self.name_scope = self.var_scope + '/'
    
    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        self._x = value

    @x.deleter
    def x(self):
        del self._x
        
    @property
    def p(self):
        return self._p

    @p.setter
    def p(self, value):
        self._p = value

    @p.deleter
    def p(self):
        del self._p

    def __init__(self, params):
        '''
        Constructor
        '''
        self._x = None
        self._p = None
        self.prefix = ''
        if 'prefix' in params:
            self.prefix = params['prefix']
        self._config_scopes()
        self._init_learning_params()
        self._cost_op = None
        