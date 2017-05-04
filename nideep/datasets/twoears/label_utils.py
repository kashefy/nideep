'''
Created on Apr 25, 2017

@author: kashefy
'''
import numpy as np
import h5py
from nideep.iow.file_system_utils import gen_paths, filter_is_h5

def id_loc_to_loc(fpath_src, key_dst, key_src='label_id_loc', has_void_bin=True):
    
    with h5py.File(fpath_src, 'r+') as h:
        if has_void_bin:
            l = np.sum(h[key_src][...,:-1], axis=1)
        else:
            l = np.sum(h[key_src], axis=1)
        l = np.expand_dims(l, 1)
        h[key_dst] = l
        
def walk_id_loc_to_loc(dir_src, key_dst):
    
    def runner(fpath):
        if filter_is_h5(fpath):
            id_loc_to_loc(fpath, key_dst)
            return True # otherwise gen_paths won't append to list
    flist = gen_paths(dir_src, func_filter=runner)
    return flist
        
if __name__ == '__main__':
    pass