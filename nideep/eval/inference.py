'''
Created on Dec 3, 2015

@author: kashefy
'''
import numpy as np
import h5py
import lmdb
import caffe
from nideep.iow import read_lmdb, to_lmdb
from nideep.iow.lmdb_utils import MAP_SZ, IDX_FMT
from nideep.blobs.mat_utils import expand_dims

def infer_to_h5_fixed_dims(net, keys, n, dst_fpath, preserve_batch=False):
    """
    Run network inference for n batches and save results to file
    """
    with h5py.File(dst_fpath, "w") as f:
        batch_sz = None
        idx_start = 0
        for itr in xrange(n):
            d = forward(net, keys)
            for k in keys:
                if preserve_batch:
                    d_copy = np.copy(d[k])
                    if k not in f:
                        shape_all = [n]
                        shape_all.extend(list(d_copy.shape))
                        f.create_dataset(k, tuple(shape_all), d_copy.dtype)
                    f[k][itr] = d_copy
                else:
                    # assume same batch size for all keys
                    if batch_sz is None:
                        batch_sz = d[k].shape[0]
                        for k2 in keys:
                            shape_all = list(d[k].shape)
                            shape_all[0] = n*batch_sz
                            f.create_dataset(k2, tuple(shape_all), d[k].dtype)
                    d_copy = np.copy(d[k])
                    f[k][idx_start:idx_start+batch_sz] = d_copy
            if not preserve_batch:
                idx_start += batch_sz
        l = [len(f[k]) for k in f.keys()]
    return l

def infer_to_lmdb(net, keys, n, dst_prefix):
    """
    Run network inference for n batches and save results to an lmdb for each key.
    Lower time complexity but much higher space complexity.
    
    Not recommended for large datasets or large number of keys
    See: infer_to_lmdb_cur() for slower alternative with less memory overhead
    
    lmdb cannot preserve batches
    """
    dc = {k:[] for k in keys}
    for _ in range(n):
        d = forward(net, keys)
        for k in keys:
            dc[k].extend(np.copy(d[k].astype(float)))

    for k in keys:
        to_lmdb.arrays_to_lmdb(dc[k], dst_prefix % (k,))

    return [len(dc[k]) for k in keys]

def infer_to_lmdb_cur(net, keys, n, dst_prefix):
    '''
    Run network inference for n batches and save results to an lmdb for each key.
    Higher time complexity but lower space complexity.
    
    Recommended for large datasets or large number of keys
    See: infer_to_lmdb() for faster alternative but with higher memory overhead
    
    lmdb cannot preserve batches
    '''
    dbs = {k : lmdb.open(dst_prefix % (k,), map_size=MAP_SZ) for k in keys}

    if len(keys) == 1:
        key_ = keys[0]
        num_written = _infer_to_lmdb_cur_single_key(net, key_, n, dbs[key_])
    else:
        num_written = _infer_to_lmdb_cur_multi_key(net, keys, n, dbs)

    for k in keys:
        dbs[k].close()

    return num_written

def _infer_to_lmdb_cur_single_key(net, key_, n, db):
    '''
    Run network inference for n batches and save results to an lmdb for each key.
    Higher time complexity but lower space complexity.
    
    Takes advantage if there is only a single key
    '''
    idx = 0

    with db.begin(write=True) as txn:
        for _ in range(n):
            d = forward(net, [key_])
            l = []
            l.extend(d[key_].astype(float))

            for x in l:
                x = expand_dims(x, 3)
                txn.put(IDX_FMT.format(idx), caffe.io.array_to_datum(x).SerializeToString())
                idx += 1
    return [idx]

def _infer_to_lmdb_cur_multi_key(net, keys, n, dbs):
    '''
    Run network inference for n batches and save results to an lmdb for each key.
    Higher time complexity but lower space complexity.
    
    See _infer_to_lmdb_cur_single_key() if there is only a single key
    '''
    idxs = [0] * len(keys)

    for _ in range(n):
        d = forward(net, keys)
        for ik, k in enumerate(keys):

            with dbs[k].begin(write=True) as txn:

                l = []
                l.extend(d[k].astype(float))

                for x in l:
                    x = expand_dims(x, 3)
                    txn.put(IDX_FMT.format(idxs[ik]), caffe.io.array_to_datum(x).SerializeToString())

                    idxs[ik] += 1
    return idxs

def forward(net, keys):
    '''
    Perform forward pass on network and extract values for a set of responses
    '''
    net.forward()
    return {k : net.blobs[k].data for k in keys}

def est_min_num_fwd_passes(fpath_net, mode_str):
    """
    if multiple source for same mode, base num_passes on last
    fpath_net -- path to network definition
    mode_str -- train or test?
    
    return
    minimum no. of forward passes to cover training set
    """
    from nideep.proto.proto_utils import Parser
    np = Parser().from_net_params_file(fpath_net)

    num_passes = 0

    for l in np.layer:
        if 'data' in l.type.lower() and mode_str.lower() in l.data_param.source.lower():
            num_entries = read_lmdb.num_entries(l.data_param.source)
            num_passes = int(num_entries / l.data_param.batch_size)
            if num_entries % l.data_param.batch_size != 0:
                print("WARNING: db size not a multiple of batch size. Adding another fwd. pass.")
                num_passes += 1
            print("%d fwd. passes with batch size %d" % (num_passes, l.data_param.batch_size))

    return num_passes

def response_to_lmdb(fpath_net,
                     fpath_weights,
                     keys,
                     dst_prefix,
                     modes=None,
                     ):
    """
    keys -- name of responses to extract. Must be valid for all requested modes
    """
    modes = modes or [caffe.TRAIN, caffe.TEST]
    out = dict.fromkeys(modes)

    for m in modes:
        num_passes = est_min_num_fwd_passes(fpath_net, ['train', 'test'][m])
        out[m] = infer_to_lmdb(caffe.Net(fpath_net, fpath_weights, m),
                               keys,
                               num_passes,
                               dst_prefix + '%s_' + ['train', 'test'][m] + '_lmdb')
    return out

if __name__ == '__main__':

#     from os.path import expanduser
#
#     fpath_net = expanduser('~/models/dark/mnist/t0/lenet_train_test_inf.prototxt')
#     fpath_weights = expanduser('~/models/dark/mnist/t0/lenet_iter_10000.caffemodel')
#     net = caffe.Net(fpath_net, fpath_weights, caffe.TRAIN)
#
#     fpath = expanduser('~/models/dark/mnist/t0/mnist_ip2_train')
#     num_entries = read_lmdb.num_entries(expanduser('~/data/mnist/mnist_train_lmdb'))
#
#     infer_to_h5_fixed_dims(net, ['ip2'], num_entries, "%s.h5" % fpath)
#     infer_to_lmdb(net, ['ip2'], num_entries, "%s_lmdb" % fpath)

    from os.path import expanduser

    fpath_net = expanduser('~/models/dark/mnist/t0/lenet_train_test.prototxt')
    fpath_weights = expanduser('~/models/dark/mnist/t0/lenet_iter_10000.caffemodel')

#     x = response_to_lmdb(fpath_net, fpath_weights,
#                      ['ip2', 'ip1'],
#                      expanduser('~/models/dark/mnist/t0/mnistX_'))

    net = caffe.Net(fpath_net, fpath_weights, caffe.TRAIN)
    keys = ['ip2', 'ip1']
    x = infer_to_lmdb_cur(net, keys, 2,
                      expanduser('~/models/dark/mnist/t0/Xmnist_%s_train_lmdb'))

    print x

    import os
    print [os.path.isdir(expanduser('~/models/dark/mnist/t0/Xmnist_%s_train_lmdb') % (k,)) for k in keys]
    print [read_lmdb.num_entries(expanduser('~/models/dark/mnist/t0/Xmnist_%s_train_lmdb') % (k,)) for k in keys]
    # print [read_lmdb.read_values(expanduser('~/models/dark/mnist/t0/Xmnist_%s_train_lmdb') % (k,)) for k in keys]

#     with h5py.File(fpath, "w") as f:
#
#         f['a'] = 0
#
#
#         f['b'] = [1, 2]
#         f['c'] = np.arange(3)
#         f['d'] = [np.array([[1,2],[4,5]], dtype=float), np.array([[1,2],[4, 5]], dtype=float)+10]

    # infer_to_h5(net, 1, ['accuracy'], fpath)
