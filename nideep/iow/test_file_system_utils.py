'''
Created on Oct 30, 2015

@author: kashefy
'''
from nose.tools import assert_false, assert_true, assert_is_instance, \
    assert_equal, assert_greater, assert_in, assert_list_equal
import os
import tempfile
import shutil
import sys
import file_system_utils as fs
from nideep.eval.log_utils import is_caffe_info_log

CURRENT_MODULE_PATH = os.path.abspath(sys.modules[__name__].__file__)
ROOT_PKG_PATH = os.path.dirname(CURRENT_MODULE_PATH)
TEST_DATA_DIRNAME = 'test_data'

def test_filter_is_img():

    assert_false(fs.filter_is_img('foo.bar'))
    assert_false(fs.filter_is_img('foo.png.bar'))
    assert_true(fs.filter_is_img('foo.png'))
    assert_true(fs.filter_is_img('foo.jpg'))

def test_filter_is_h5():

    assert_false(fs.filter_is_h5('foo.bar'))
    assert_false(fs.filter_is_h5('foo.h5.bar'))
    assert_false(fs.filter_is_h5('foo.hdf'))
    assert_false(fs.filter_is_h5('foo.h4'))
    assert_false(fs.filter_is_h5('foo.hdf4'))
    assert_true(fs.filter_is_h5('foo.h5'))
    assert_true(fs.filter_is_h5('foo.hdf5'))

class TestFSUtils:

    @classmethod
    def setup_class(self):

        self.dir_ = os.path.join(os.path.dirname(ROOT_PKG_PATH),
                                          TEST_DATA_DIRNAME)

    def test_gen_paths_no_filter(self):

        flist = fs.gen_paths(self.dir_)
        assert_is_instance(flist, list)
        assert_greater(len(flist), 0)

    def test_gen_paths_is_caffe_log(self):

        flist = fs.gen_paths(self.dir_, is_caffe_info_log)
        assert_is_instance(flist, list)
        assert_equal(len(flist), 1)
        assert_true('.log.' in flist[0] and '.INFO.' in flist[0])

    def test_gen_paths_no_imgs_found(self):

        flist = fs.gen_paths(self.dir_, fs.filter_is_img)
        assert_equal(len(flist), 0)

    def test_hash_file(self):

        p = fs.gen_paths(self.dir_, is_caffe_info_log)[0]
        h = fs.hashfile(p)
        assert_is_instance(h, str)
        assert_greater(len(h), 0)

class TestFNamePairs:

    @classmethod
    def setup_class(self):

        self.dir_tmp = tempfile.mkdtemp()
        self.dir_ = os.path.join(os.path.dirname(ROOT_PKG_PATH),
                                          TEST_DATA_DIRNAME)
        shutil.copytree(self.dir_, os.path.join(self.dir_tmp, 'subdir'))

    @classmethod
    def teardown_class(self):

        shutil.rmtree(self.dir_tmp)

    def test_fname_pairs(self):

        a = ['foo1_a.txt', os.path.join('foo', 'bar_x.txt'), 'foo5.txt']
        b = [os.path.join('oof', 'bar_x.txt'), 'foo5_b.txt', 'foo2_b.txt']
        pairs = fs.fname_pairs(a, b)

        for x, y in pairs:
            assert_in(x, a)
            assert_in(y, b)

        assert_list_equal(pairs, [[os.path.join('foo', 'bar_x.txt'),
                                   os.path.join('oof', 'bar_x.txt')],
                                  ['foo5.txt', 'foo5_b.txt'],
                                  ])

    def test_fname_pairs_log_files(self):

        a = fs.gen_paths(self.dir_, is_caffe_info_log)
        b = fs.gen_paths(self.dir_tmp, is_caffe_info_log)
        pairs = fs.fname_pairs(a, b)

        for x, y in pairs:
            assert_in(x, a)
            assert_in(y, b)


