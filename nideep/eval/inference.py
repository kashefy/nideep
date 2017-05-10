'''
Created on Dec 3, 2015

@author: kashefy
'''
import logging
import numpy as np
import h5py
import lmdb
import caffe
from nideep.iow import to_lmdb
from nideep.iow.dataSource import CreateDatasource
from nideep.iow.lmdb_utils import MAP_SZ, IDX_FMT
from nideep.blobs.mat_utils import expand_dims

logger = logging.getLogger(__name__)

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
                            shape_all = list(d[k2].shape)
                            shape_all[0] = n * batch_sz
                            f.create_dataset(k2, tuple(shape_all), d[k].dtype)
                    d_copy = np.copy(d[k])
                    f[k][idx_start:idx_start + batch_sz] = d_copy
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

def est_min_num_fwd_passes(fpath_net, mode_str, key=None):
    """
    if multiple source for same mode, base num_passes on last
    fpath_net -- path to network definition
    mode_str -- train or test?

    return
    minimum no. of forward passes to cover training set
    """
    from nideep.proto.proto_utils import Parser
    mode_num = {'train' : caffe.TRAIN,
                'test' : caffe.TEST}[mode_str]
    np = Parser().from_net_params_file(fpath_net)
    num_passes_each = []
    for l in np.layer:
        if 'data' in l.type.lower():
            if ('hdf5data' in l.type.lower() and
                    (mode_str.lower() in l.hdf5_data_param.source.lower() or
                        [x.phase for x in l.include] == [mode_num])):
                num_entries = CreateDatasource.from_path(l.hdf5_data_param.source, key=key).num_entries()
                num_passes = int(num_entries / l.hdf5_data_param.batch_size)
                if num_entries % l.hdf5_data_param.batch_size != 0:
                    logger.warning("db size not a multiple of batch size. Adding another fwd. pass.")
                    num_passes += 1
                logger.info("%d fwd. passes with batch size %d" % (num_passes, l.hdf5_data_param.batch_size))
                num_passes_each.append(num_passes)
            elif (mode_str.lower() in l.data_param.source.lower() or
                    [x.phase for x in l.include] == [mode_num]):
                num_entries = CreateDatasource.from_path(l.data_param.source, key=key).num_entries()
                num_passes = int(num_entries / l.data_param.batch_size)
                if num_entries % l.data_param.batch_size != 0:
                    logger.warning("db size not a multiple of batch size. Adding another fwd. pass.")
                    num_passes += 1
                logger.info("%d fwd. passes with batch size %d" % (num_passes, l.data_param.batch_size))
                num_passes_each.append(num_passes)
    return max(num_passes_each)

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
    pass
