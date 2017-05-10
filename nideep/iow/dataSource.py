'''
Created on May 5, 2017

@author: kashefy
'''
import os
import logging
from abc import abstractmethod, ABCMeta
import h5py
import lmdb
from nideep.iow import read_lmdb

class CreateDatasource(object):
    """ Factory method for instantiating appropriate data source """
    @classmethod
    def from_path(cls, p, key=None):
        if os.path.splitext(p)[-1] == '.txt':
            return DataSourceH5List(p, key)
        if p.endswith('lmdb'):
            return DataSourceLMDB(p)
        raise ValueError("Unrecognized data source (%s)" % p)

class AbstractDataSource(object):
    '''
    classdocs
    '''
    __metaclass__ = ABCMeta

    @abstractmethod
    def num_entries(self):
        pass

    @abstractmethod
    def exists(self):
        pass

    def __init__(self, p):
        '''
        Constructor
        '''
        self.p = p
        self.logger = logging.getLogger(__name__)
        self.exists()

class DataSourceLMDB(AbstractDataSource):
    '''
    classdocs
    '''
    def num_entries(self):
        return read_lmdb.num_entries(self.p)

    def exists(self):
        if not os.path.isdir(self.p):
            raise lmdb.Error("LMDB not found (%s)")

    def __init__(self, p):
        '''
        Constructor
        '''
        super(DataSourceLMDB, self).__init__(p)

class DataSourceH5(AbstractDataSource):
    '''
    classdocs
    '''
    def num_entries(self):
        with h5py.File(self.p, 'r') as h:
            return len(h[self.key])

    def exists(self):
        with h5py.File(self.p, 'r') as h:
            if self.key not in h.keys():
                raise KeyError("Could not find %s in HDF5 file %s", self.p)

    def __init__(self, p, key):
        '''
        Constructor
        '''
        self.key = key
        super(DataSourceH5, self).__init__(p)

class DataSourceH5List(AbstractDataSource):
    '''
    classdocs
    '''
    def num_entries(self):
        num_total = 0
        for h in self.h5list:
            num_total += h.num_entries()
        return num_total

    def exists(self):
        with open(self.p, 'r') as f:
            for l in f:
                self.h5list.append(DataSourceH5(l.rstrip('\n'), self.key))

    def __init__(self, p, key):
        '''
        Constructor
        '''
        self.key = key
        self.h5list = []
        super(DataSourceH5List, self).__init__(p)
