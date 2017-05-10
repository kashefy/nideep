'''
Created on Oct 30, 2015

@author: kashefy
'''
from nose.tools import assert_equal, assert_false, \
    assert_list_equal, assert_true
from mock import patch, PropertyMock
import os
import tempfile
import shutil
import numpy as np
from numpy.testing import assert_array_equal
import h5py
import inference as infr
import sys
CURRENT_MODULE_PATH = os.path.abspath(sys.modules[__name__].__file__)
ROOT_PKG_PATH = os.path.dirname(os.path.dirname(CURRENT_MODULE_PATH))
TEST_DATA_DIRNAME = 'test_data'
TEST_NET_FILENAME = 'n1.prototxt'
TEST_NET_HDF5DATA_FILENAME = 'n1h.prototxt'

class Bunch:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)

class TestInference:

    @patch('nideep.eval.inference.caffe.Net')
    def test_forward(self, mock_net):

        # fake minimal test data
        b = {k : Bunch(data=np.random.rand(3, 2)) for k in ['x', 'y', 'z']}

        # mock methods and properties of Net objects
        mock_net.return_value.forward.return_value = np.zeros(1)
        type(mock_net.return_value).blobs = PropertyMock(return_value=b)
        net = mock_net()

        assert_false(net.forward.called, "Problem with mocked forward()")

        out = infr.forward(net, ['x', 'z'])

        assert_true(net.forward.called, "Problem with mocked forward()")

        assert_list_equal(out.keys(), ['x', 'z'])
        for k in ['x', 'z']:
            assert_equal(out[k].shape, (3, 2),
                         msg="unexpected shape for blob %s" % k)

            assert_array_equal(b[k].data, out[k])

        # repeat with smaller set of keys
        out = infr.forward(net, ['z'])
        assert_list_equal(out.keys(), ['z'])
        assert_equal(out['z'].shape, (3, 2), msg="unexpected shape for blob z")
        assert_array_equal(b['z'].data, out['z'])

class TestInferenceEstNumFwdPasses():

    @patch('nideep.iow.dataSource.DataSourceLMDB')
    def test_est_num_fwd_passes_caffe_lmdb(self, mock_ds):

        # we know the batch sizes from the prototxt file
        fpath_net = os.path.join(ROOT_PKG_PATH, TEST_DATA_DIRNAME, TEST_NET_FILENAME)

        mock_ds.return_value.num_entries.return_value = 77 * 64  # got batch size 64 from files directly
        assert_equal(77, infr.est_min_num_fwd_passes(fpath_net, 'train'))

        mock_ds.return_value.num_entries.return_value = 33 * 100  # got batch size 64 from files directly
        fpath_net = os.path.join(ROOT_PKG_PATH, TEST_DATA_DIRNAME, TEST_NET_FILENAME)
        assert_equal(33, infr.est_min_num_fwd_passes(fpath_net, 'test'))

    @patch('nideep.iow.dataSource.DataSourceH5List')
    def test_est_num_fwd_passes_caffe_h5list(self, mock_ds):

        # we know the batch sizes from the prototxt file
        fpath_net = os.path.join(ROOT_PKG_PATH, TEST_DATA_DIRNAME, TEST_NET_HDF5DATA_FILENAME)

        mock_ds.return_value.num_entries.return_value = 44 * 64  # got batch size 64 from files directly
        assert_equal(44, infr.est_min_num_fwd_passes(fpath_net, 'train'))

        mock_ds.return_value.num_entries.return_value = 11 * 128  # got batch size 64 from files directly
        fpath_net = os.path.join(ROOT_PKG_PATH, TEST_DATA_DIRNAME, TEST_NET_HDF5DATA_FILENAME)
        assert_equal(11, infr.est_min_num_fwd_passes(fpath_net, 'test'))

class TestInferenceHDF5:

    @classmethod
    def setup_class(self):

        self.dir_tmp = tempfile.mkdtemp()

    @classmethod
    def teardown_class(self):

        shutil.rmtree(self.dir_tmp)

    @patch('nideep.eval.inference.caffe.Net')
    def test_infer_to_h5_fixed_dims(self, mock_net):

        # fake minimal test data
        b = {k : Bunch(data=np.random.rand(1, 1 + idx, 3, 2 * (idx + 1))) for idx, k in enumerate(['x', 'y', 'z'])}

        # mock methods and properties of Net objects
        mock_net.return_value.forward.return_value = np.zeros(1)
        type(mock_net.return_value).blobs = PropertyMock(return_value=b)
        net = mock_net()

        fpath = os.path.join(self.dir_tmp, 'test_infer_to_h5_fixed_dims.h5')
        assert_false(os.path.isfile(fpath))

        out = infr.infer_to_h5_fixed_dims(net, ['x', 'z'], 1, fpath)

        assert_equal(net.forward.call_count, 1)
        assert_true(os.path.isfile(fpath))
        assert_list_equal(out, [1, 1])

        # check db content
        with h5py.File(fpath, "r") as f:
            assert_list_equal([str(k) for k in f.keys()], ['x', 'z'])

            for idx, k in enumerate(['x', 'y', 'z']):
                if k == 'y':
                    assert_false(k in f, "Unexpected key found (%s)" % k)
                else:
                    assert_equal(f[k].shape, (1, 1 + idx, 3, 2 * (idx + 1)),
                                 msg="unexpected shape for blob %s" % k)
            assert_array_equal(b[k].data, f[k])


    @patch('nideep.eval.inference.caffe.Net')
    def test_infer_to_h5_fixed_dims_n(self, mock_net):

        # fake minimal test data
        b = {k : Bunch(data=np.random.rand(1, 1, 3, 2)) for k in ['x', 'y', 'z']}

        # mock methods and properties of Net objects
        mock_net.return_value.forward.return_value = np.zeros(1)
        type(mock_net.return_value).blobs = PropertyMock(return_value=b)

        for n in range(1, 10):

            net = mock_net()
            net.reset_mock()
            fpath = os.path.join(self.dir_tmp, 'test_infer_to_h5_fixed_dims_n.h5')
            out = infr.infer_to_h5_fixed_dims(net, ['x', 'z'], n, fpath)

            assert_equal(net.forward.call_count, n)
            assert_list_equal(out, [n, n])

    @patch('nideep.eval.inference.caffe.Net')
    def test_infer_to_h5_fixed_dims_preserve_batch_no(self, mock_net):

        # fake minimal test data
        b = {k : Bunch(data=np.random.rand(4, 1, 3, 2)) for k in ['x', 'y', 'z']}

        # mock methods and properties of Net objects
        mock_net.return_value.forward.return_value = np.zeros(1)
        type(mock_net.return_value).blobs = PropertyMock(return_value=b)
        net = mock_net()

        fpath = os.path.join(self.dir_tmp, 'test_infer_to_h5_fixed_dims_preserve_batch_no.h5')
        assert_false(os.path.isfile(fpath))

        n = 3
        out = infr.infer_to_h5_fixed_dims(net, ['x', 'z'], n, fpath,
                                          preserve_batch=False)

        assert_equal(net.forward.call_count, n)
        assert_true(os.path.isfile(fpath))
        assert_list_equal(out, [n * 4] * 2)

    @patch('nideep.eval.inference.caffe.Net')
    def test_infer_to_h5_fixed_dims_preserve_batch_yes(self, mock_net):

        # fake minimal test data
        b = {k : Bunch(data=np.random.rand(4, 1, 3, 2)) for k in ['x', 'y', 'z']}

        # mock methods and properties of Net objects
        mock_net.return_value.forward.return_value = np.zeros(1)
        type(mock_net.return_value).blobs = PropertyMock(return_value=b)
        net = mock_net()

        fpath = os.path.join(self.dir_tmp, 'test_infer_to_h5_fixed_dims_preserve_batch_yes.h5')
        assert_false(os.path.isfile(fpath))

        n = 3
        out = infr.infer_to_h5_fixed_dims(net, ['x', 'z'], n, fpath,
                                          preserve_batch=True)

        assert_equal(net.forward.call_count, n)
        assert_true(os.path.isfile(fpath))
        assert_list_equal(out, [n] * 2)

class TestInferenceLMDB:

    @classmethod
    def setup_class(self):

        self.dir_tmp = tempfile.mkdtemp()

    @classmethod
    def teardown_class(self):

        shutil.rmtree(self.dir_tmp)

    @patch('nideep.eval.inference.caffe.Net')
    def test_infer_to_lmdb_fixed_dims(self, mock_net):

        # fake minimal test data
        b = {k : Bunch(data=np.random.rand(1, 1, 3, 2)) for k in ['x', 'y', 'z']}

        # mock methods and properties of Net objects
        mock_net.return_value.forward.return_value = np.zeros(1)
        type(mock_net.return_value).blobs = PropertyMock(return_value=b)
        net = mock_net()

        dst_prefix = os.path.join(self.dir_tmp, 'test_infer_to_lmdb_fixed_dims_%s_lmdb')
        for k in b.keys():
            assert_false(os.path.isdir(dst_prefix % k))

        out = infr.infer_to_lmdb(net, ['x', 'z'], 1, dst_prefix)

        assert_equal(net.forward.call_count, 1)
        assert_list_equal(out, [1, 1])

        for k in b.keys():
            if k in ['x', 'z']:
                assert_true(os.path.isdir(dst_prefix % k))
            else:
                assert_false(os.path.isdir(dst_prefix % k))

    @patch('nideep.eval.inference.caffe.Net')
    def test_infer_to_lmdb_fixed_dims_n(self, mock_net):

        # fake minimal test data
        b = {k : Bunch(data=np.random.rand(1, 1, 3, 2)) for k in ['x', 'y', 'z']}

        # mock methods and properties of Net objects
        mock_net.return_value.forward.return_value = np.zeros(1)
        type(mock_net.return_value).blobs = PropertyMock(return_value=b)

        for n in range(1, 10):

            net = mock_net()
            net.reset_mock()

            dst_prefix = os.path.join(self.dir_tmp, 'test_infer_to_lmdb_fixed_dims_n_%s_lmdb')
            out = infr.infer_to_lmdb(net, ['x', 'z'], n, dst_prefix)

            assert_equal(net.forward.call_count, n)
            assert_list_equal(out, [n, n])

    @patch('nideep.eval.inference.caffe.Net')
    def test_infer_to_lmdb_fixed_dims_preserve_batch_no(self, mock_net):

        # fake minimal test data
        b = {k : Bunch(data=np.random.rand(4, 1, 3, 2)) for k in ['x', 'y', 'z']}

        # mock methods and properties of Net objects
        mock_net.return_value.forward.return_value = np.zeros(1)
        type(mock_net.return_value).blobs = PropertyMock(return_value=b)
        net = mock_net()

        dst_prefix = os.path.join(self.dir_tmp, 'test_infer_to_lmdb_fixed_dims_preserve_batch_no_%s_lmdb')
        for k in b.keys():
            assert_false(os.path.isdir(dst_prefix % k))

        n = 3
        out = infr.infer_to_lmdb(net, ['x', 'z'], n, dst_prefix)

        assert_equal(net.forward.call_count, n)
        assert_list_equal(out, [n * 4] * 2)
        for k in b.keys():
            if k in ['x', 'z']:
                assert_true(os.path.isdir(dst_prefix % k))
            else:
                assert_false(os.path.isdir(dst_prefix % k))

    @patch('nideep.eval.inference.caffe.Net')
    def test_infer_to_lmdb_cur_multi_key(self, mock_net):

        # fake minimal test data
        b = {k : Bunch(data=np.random.rand(4, 1, 3, 2)) for k in ['x', 'y', 'z']}

        # mock methods and properties of Net objects
        mock_net.return_value.forward.return_value = np.zeros(1)
        type(mock_net.return_value).blobs = PropertyMock(return_value=b)
        net = mock_net()

        dst_prefix = os.path.join(self.dir_tmp, 'test_infer_to_lmdb_cur_multi_key_%s_lmdb')
        for k in b.keys():
            assert_false(os.path.isdir(dst_prefix % k))

        n = 3
        out = infr.infer_to_lmdb_cur(net, ['x', 'z'], n, dst_prefix)

        assert_equal(net.forward.call_count, n)
        assert_list_equal(out, [n * 4] * 2)
        for k in b.keys():
            if k in ['x', 'z']:
                assert_true(os.path.isdir(dst_prefix % k))
            else:
                assert_false(os.path.isdir(dst_prefix % k))

    @patch('nideep.eval.inference.caffe.Net')
    def test_infer_to_lmdb_cur_single_key(self, mock_net):

        # fake minimal test data
        b = {k : Bunch(data=np.random.rand(4, 1, 3, 2)) for k in ['x', 'y', 'z']}

        # mock methods and properties of Net objects
        mock_net.return_value.forward.return_value = np.zeros(1)
        type(mock_net.return_value).blobs = PropertyMock(return_value=b)
        net = mock_net()

        dst_prefix = os.path.join(self.dir_tmp, 'test_infer_to_lmdb_cur_single_key_%s_lmdb')
        for k in b.keys():
            assert_false(os.path.isdir(dst_prefix % k))

        n = 3
        out = infr.infer_to_lmdb_cur(net, ['z'], n, dst_prefix)

        assert_equal(net.forward.call_count, n)
        assert_list_equal(out, [n * 4])
        for k in b.keys():
            if k in ['z']:
                assert_true(os.path.isdir(dst_prefix % k))
            else:
                assert_false(os.path.isdir(dst_prefix % k))

    @patch('nideep.eval.inference.est_min_num_fwd_passes')
    @patch('nideep.eval.inference.caffe.Net')
    def test_response_to_lmdb(self, mock_net, mock_num):

        # fake minimal test data
        b = {k : Bunch(data=np.random.rand(4, 1, 3, 2)) for k in ['x', 'y', 'z']}

        # mock methods and properties of Net objects
        mock_num.return_value = 3
        mock_net.return_value.forward.return_value = np.zeros(1)
        type(mock_net.return_value).blobs = PropertyMock(return_value=b)
        net = mock_net()

        dst_prefix = os.path.join(self.dir_tmp, 'test_response_to_lmdb_')
        for m in ['train', 'test']:
            for k in b.keys():
                assert_false(os.path.isdir(dst_prefix + ('%s_' + m + '_lmdb') % k))
        import nideep
        out = nideep.eval.inference.response_to_lmdb("net.prototxt",
                                                     "w.caffemodel",
                                                     ['x', 'z'],
                                                     dst_prefix)

        assert_equal(net.forward.call_count, 3 * 2)  # double for both modes
        from caffe import TRAIN, TEST
        assert_list_equal(out.keys(), [TRAIN, TEST])
        assert_list_equal(out[TRAIN], [3 * 4] * 2)
        assert_list_equal(out[TEST], [3 * 4] * 2)

        for m in ['train', 'test']:
            for k in b.keys():
                if k in ['x', 'z']:
                    assert_true(os.path.isdir(dst_prefix + ('%s_' + m + '_lmdb') % k))
                else:
                    assert_false(os.path.isdir(dst_prefix + ('%s_' + m + '_lmdb') % k))

