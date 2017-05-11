'''
Created on May 5, 2017

@author: kashefy
'''
import os
import random
import string
import shutil
import tempfile
from nose.tools import assert_equals, assert_false, \
    assert_raises, assert_true, assert_is_instance
import numpy as np
import lmdb
import h5py
import dataSource as ds

def create_empty_lmdb(p):
    db = lmdb.open(p, map_size=int(1e12))
    with db.begin(write=True) as _:
        _  # do nothing
    db.close()

class TestCreateDatasource:

    def setup(self):
        self.dir_tmp = tempfile.mkdtemp()

    def teardown(self):
        shutil.rmtree(self.dir_tmp)

    def test_from_path_lmdb(self):

        for fname in ['x_lmdb', 'x.lmdb']:
            fpath = os.path.join(self.dir_tmp, fname)
            create_empty_lmdb(fpath)
            assert_is_instance(ds.CreateDatasource.from_path(fpath), ds.DataSourceLMDB)

    def test_from_path_h5list(self):
        for fname in ['foo.txt']:
            fpath = os.path.join(self.dir_tmp, fname)
            with open(fpath, 'a') as f:
                path_h5 = os.path.join(self.dir_tmp, 'x.h5')
                f.write(path_h5 + '\n')
            with h5py.File(path_h5, 'w') as h:
                h['a'] = np.random.rand(10, 20)
                h['b'] = np.random.rand(20, 10, 20)
            assert_is_instance(ds.CreateDatasource.from_path(fpath, 'a'), ds.DataSourceH5List)

    def test_from_path_invalid(self):
        for fname in ['foo.png', 'foo.abc']:
            fpath = os.path.join(self.dir_tmp, fname)
            with open(fpath, 'wb') as f:
                f.write("bla")
            assert_raises(ValueError, ds.CreateDatasource.from_path, fpath)

class TestDataSourceLMDB:
    @staticmethod
    def randtxt():
        return ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(9))

    def setup(self):
        self.dir_tmp = tempfile.mkdtemp()

    def teardown(self):
        shutil.rmtree(self.dir_tmp)

    def test_num_entries(self):

        for n in xrange(19):
            path_lmdb = os.path.join(self.dir_tmp, 'x_lmdb')
            db = lmdb.open(path_lmdb, map_size=int(1e12))
            with db.begin(write=True) as in_txn:
                for idx in xrange(n):
                    in_txn.put('{:0>10d}'.format(idx), TestDataSourceLMDB.randtxt())
            db.close()
            assert_equals(n, ds.DataSourceLMDB(path_lmdb).num_entries())

    def test_empty(self):

        path_lmdb_empty = os.path.join(self.dir_tmp, "empty_lmdb")
        create_empty_lmdb(path_lmdb_empty)
        assert_true(os.path.isdir(path_lmdb_empty), "Empty LMDB does not exist")
        assert_equals(0, ds.DataSourceLMDB(path_lmdb_empty).num_entries())

    def test_does_not_exist(self):

        path_lmdb = os.path.join(self.dir_tmp, 'test_num_entries_does_not_exist_lmdb')
        assert_false(os.path.exists(path_lmdb))
        assert_raises(lmdb.Error, ds.DataSourceLMDB, path_lmdb)

class TestDataSourceH5List:

    def setup(self):
        self.dir_tmp = tempfile.mkdtemp()

    def teardown(self):
        shutil.rmtree(self.dir_tmp)

    def test_num_entries(self):

        path_list = os.path.join(self.dir_tmp, 'x.txt')
        n_running = 0
        for n in xrange(1, 19):
            with open(path_list, 'a') as f:
                n_running += n
                path_h5 = os.path.join(self.dir_tmp, 'x%d.h5' % n)
                f.write(path_h5 + '\n')
            with h5py.File(path_h5, 'w') as h:
                h['x'] = np.random.rand(n, 10, 20)
                h['y'] = np.random.rand(n + 10, 10, 20)
            assert_equals(n, ds.DataSourceH5(path_h5, 'x').num_entries())
            assert_equals(n + 10, ds.DataSourceH5(path_h5, 'y').num_entries())
            assert_equals(n_running, ds.DataSourceH5List(path_list, 'x').num_entries())

    def test_empty(self):

        path_list = os.path.join(self.dir_tmp, 'x.txt')
        open(path_list, 'w')
        assert_equals(0, ds.DataSourceH5List(path_list, 'x').num_entries())

    def test_key_does_not_exist(self):

        path_list = os.path.join(self.dir_tmp, 'x.txt')
        with open(path_list, 'a') as f:
            path_h5 = os.path.join(self.dir_tmp, 'x.h5')
            f.write(path_h5 + '\n')
        with h5py.File(path_h5, 'w') as h:
            h['x'] = np.random.rand(10, 10, 20)
            h['y'] = np.random.rand(20, 10, 20)
        assert_raises(KeyError, ds.DataSourceH5List, path_list, 'z')
        assert_raises(KeyError, ds.DataSourceH5, path_h5, 'z')

    def test_does_not_exist(self):

        path_list = os.path.join(self.dir_tmp, 'does_not_exist.txt')
        assert_false(os.path.exists(path_list))
        assert_raises(IOError, ds.DataSourceH5List, path_list, 'x')

    def test_does_not_exist_content(self):

        path_list = os.path.join(self.dir_tmp, 'x.txt')
        with open(path_list, 'a') as f:
            path_h5 = os.path.join(self.dir_tmp, 'does_not_exist.h5')
            f.write(path_h5 + '\n')
        assert_true(os.path.exists(path_list))
        assert_false(os.path.exists(path_h5))
        assert_raises(IOError, ds.DataSourceH5List, path_list, 'x')
