'''
Created on Oct 30, 2015

@author: kashefy
'''
from nose.tools import assert_equal, assert_false, \
    assert_not_equal, assert_list_equal, assert_raises, raises
from mock import patch, PropertyMock
import os
import tempfile
import shutil
import numpy as np
import lmdb
import caffe
import read_lmdb as r

class TestReadLabelsLMDB:

    @classmethod
    def setup_class(self):

        self.dir_tmp = tempfile.mkdtemp()
        self.path_lmdb = os.path.join(self.dir_tmp, 'x_lmdb')

        # write fake data to lmdb
        db = lmdb.open(self.path_lmdb, map_size=int(1e12))
        with db.begin(write=True) as in_txn:
            for idx in range(5):
                in_txn.put('{:0>10d}'.format(idx), "%s" % (idx * 10))
        db.close()

    @classmethod
    def teardown_class(self):

        shutil.rmtree(self.dir_tmp)

    @patch('nideep.iow.read_lmdb.caffe.proto.caffe_pb2.Datum')
    def test_read_labels(self, mock_dat):

        # mock methods and properties of Datum objects
        mock_dat.return_value.ParseFromString.return_value = ""
        type(mock_dat.return_value).label = PropertyMock(side_effect=range(5))

        assert_list_equal(r.read_labels(self.path_lmdb), range(5))

class TestReadValuesNoLabelLMDB:

    def setup(self):

        self.dir_tmp = tempfile.mkdtemp()

        self.img1_data = np.array([[[ 1, 2, 3],
                                    [ 4, 5, 6]
                                    ],
                                   [[ 7, 8, 9],
                                    [10, 11, 12]
                                    ],
                                   [[13, 14, 15],
                                    [16, 17, 18],
                                    ],
                                   [[19, 20, 21],
                                    [22, 23, 24]
                                    ]
                                   ])

        img_data_str = ['\x08\x03\x10\x04\x18\x02"\x18\x01\x04\x07\n\r\x10\x13\x16\x02\x05\x08\x0b\x0e\x11\x14\x17\x03\x06\t\x0c\x0f\x12\x15\x18(\x00',
                        '\x08\x03\x10\x02\x18\x01"\x06\x10\x16\x11\x17\x12\x18(\x00']

        # write fake data to lmdb
        self.path_lmdb = os.path.join(self.dir_tmp, 'imgs_lmdb')
        db = lmdb.open(self.path_lmdb, map_size=int(1e12))
        with db.begin(write=True) as in_txn:

            for idx, data_str in enumerate(img_data_str):
                in_txn.put('{:0>10d}'.format(idx), data_str)
        db.close()

    def teardown(self):

        shutil.rmtree(self.dir_tmp)

    @patch('nideep.iow.read_lmdb.caffe.proto.caffe_pb2.Datum')
    def test_read_values_pixels_no_labels(self, mock_dat):

        # expected content: img1_data, img1_data[2:,1:,:]

        # mocks
        dstr = ['\x01\x04\x07\n\r\x10\x13\x16\x02\x05\x08\x0b\x0e\x11\x14\x17\x03\x06\t\x0c\x0f\x12\x15\x18',
                '\x10\x16\x11\x17\x12\x18']

        mock_dat.return_value.ParseFromString.return_value = ""
        type(mock_dat.return_value).data = PropertyMock(side_effect=dstr)
        type(mock_dat.return_value).channels = PropertyMock(side_effect=[3, 3])
        type(mock_dat.return_value).height = PropertyMock(side_effect=[4, 2])
        type(mock_dat.return_value).width = PropertyMock(side_effect=[2, 1])
        type(mock_dat.return_value).label = PropertyMock(return_value=0)

        v = r.read_values(self.path_lmdb)

        assert_equal(len(v), 2)

        x, l = v[0]
        for ch in range(3):
            for row in range(4):
                for col in range(2):
                    assert_equal(x[ch, row, col], self.img1_data[row, col, ch])

        assert_equal(l, 0, "Unexpected 1st label")

        x, l = v[1]
        img2_data = self.img1_data[2:, 1:, :]

        for ch in range(3):
            for row in range(2):
                for col in range(1):
                    assert_equal(x[ch, row, col], img2_data[row, col, ch])

        assert_equal(l, 0, "Unexpected 2nd label")

class TestReadValuesWithLabelLMDB:

    def setup(self):

        self.dir_tmp = tempfile.mkdtemp()

        self.img1_data = np.array([[[ 1, 2, 3],
                                    [ 4, 5, 6]
                                    ],
                                   [[ 7, 8, 9],
                                    [10, 11, 12]
                                    ],
                                   [[13, 14, 15],
                                    [16, 17, 18],
                                    ],
                                   [[19, 20, 21],
                                    [22, 23, 24]
                                    ]
                                   ])

        img_data_str = ['\x08\x03\x10\x04\x18\x02"\x18\x01\x04\x07\n\r\x10\x13\x16\x02\x05\x08\x0b\x0e\x11\x14\x17\x03\x06\t\x0c\x0f\x12\x15\x18(\x01',
                        '\x08\x03\x10\x02\x18\x01"\x06\x10\x16\x11\x17\x12\x18(\x00']

        # write fake data to lmdb
        self.path_lmdb = os.path.join(self.dir_tmp, 'imgs_lmdb')
        db = lmdb.open(self.path_lmdb, map_size=int(1e12))
        with db.begin(write=True) as in_txn:

            for idx, data_str in enumerate(img_data_str):
                in_txn.put('{:0>10d}'.format(idx), data_str)
        db.close()

    def teardown(self):

        shutil.rmtree(self.dir_tmp)

    @patch('nideep.iow.read_lmdb.caffe.proto.caffe_pb2.Datum')
    def test_read_values_pixels_with_labels(self, mock_dat):

        # mocks
        dstr = ['\x01\x04\x07\n\r\x10\x13\x16\x02\x05\x08\x0b\x0e\x11\x14\x17\x03\x06\t\x0c\x0f\x12\x15\x18',
                '\x10\x16\x11\x17\x12\x18']

        mock_dat.return_value.ParseFromString.return_value = ""
        type(mock_dat.return_value).data = PropertyMock(side_effect=dstr)
        type(mock_dat.return_value).channels = PropertyMock(side_effect=[3, 3])
        type(mock_dat.return_value).height = PropertyMock(side_effect=[4, 2])
        type(mock_dat.return_value).width = PropertyMock(side_effect=[2, 1])
        type(mock_dat.return_value).label = PropertyMock(side_effect=[1, 0])

        v = r.read_values(self.path_lmdb)

        assert_equal(len(v), 2)

        x, l = v[0]
        for ch in range(3):
            for row in range(4):
                for col in range(2):
                    assert_equal(x[ch, row, col], self.img1_data[row, col, ch])

        assert_equal(l, 1, "Unexpected 1st label")

        x, l = v[1]
        img2_data = self.img1_data[2:, 1:, :]

        for ch in range(3):
            for row in range(2):
                for col in range(1):
                    assert_equal(x[ch, row, col], img2_data[row, col, ch])

        assert_equal(l, 0, "Unexpected 2nd label")

    @patch('nideep.iow.read_lmdb.caffe.proto.caffe_pb2.Datum')
    def test_read_values_pixels_with_labels_uint8(self, mock_dat):

        # mocks
        dstr = ['\x01\x04\x07\n\r\x10\x13\x16\x02\x05\x08\x0b\x0e\x11\x14\x17\x03\x06\t\x0c\x0f\x12\x15\x18',
                '\x10\x16\x11\x17\x12\x18']

        mock_dat.return_value.ParseFromString.return_value = ""
        type(mock_dat.return_value).data = PropertyMock(side_effect=dstr)
        type(mock_dat.return_value).channels = PropertyMock(side_effect=[3, 3])
        type(mock_dat.return_value).height = PropertyMock(side_effect=[4, 2])
        type(mock_dat.return_value).width = PropertyMock(side_effect=[2, 1])
        type(mock_dat.return_value).label = PropertyMock(side_effect=[1, 0])

        v = r.read_values(self.path_lmdb, np.uint8)

        assert_equal(len(v), 2)

        x, l = v[0]
        for ch in range(3):
            for row in range(4):
                for col in range(2):
                    assert_equal(x[ch, row, col], self.img1_data[row, col, ch])

        assert_equal(l, 1, "Unexpected 1st label")

        x, l = v[1]
        img2_data = self.img1_data[2:, 1:, :]

        for ch in range(3):
            for row in range(2):
                for col in range(1):
                    assert_equal(x[ch, row, col], img2_data[row, col, ch])

        assert_equal(l, 0, "Unexpected 2nd label")

    # to generate mock data
#     def test_gen_img_lmdb(self):
#
#         import to_lmdb
#         import cv2
#         self.img1 = np.array([[[ 1,  2,  3],
#                                [ 4,  5,  6]
#                                ],
#                               [[ 7,  8,  9],
#                                [10, 11, 12]
#                                ],
#                               [[13, 14, 15],
#                                [16, 17, 18],
#                                ],
#                               [[19, 20, 21],
#                                [22, 23, 24]
#                                ]
#                               ])
#
#         print self.img1.shape
#         path_img1 = os.path.join(self.dir_tmp, "a.png")
#         cv2.imwrite(path_img1, self.img1)
#
#         path_img2 = os.path.join(self.dir_tmp, "b.png")
#         cv2.imwrite(path_img2, self.img1[2:,1:,:])
#
#         print "-------"
#         path_lmdb = os.path.join(self.dir_tmp, 'imgsX1_lmdb')
#         to_lmdb.imgs_to_lmdb([path_img1, path_img2], path_lmdb)
#
#         v = r.read_values(path_lmdb)
#         print v
#         print "-------"

    @patch('nideep.iow.read_lmdb.caffe.proto.caffe_pb2.Datum')
    def test_read_values_at(self, mock_dat):

        # mocks
        dstr = ['\x01\x04\x07\n\r\x10\x13\x16\x02\x05\x08\x0b\x0e\x11\x14\x17\x03\x06\t\x0c\x0f\x12\x15\x18',
                '\x10\x16\x11\x17\x12\x18']

        mock_dat.return_value.ParseFromString.return_value = ""
        type(mock_dat.return_value).data = PropertyMock(side_effect=dstr)
        type(mock_dat.return_value).channels = PropertyMock(side_effect=[3, 3])
        type(mock_dat.return_value).height = PropertyMock(side_effect=[4, 2])
        type(mock_dat.return_value).width = PropertyMock(side_effect=[2, 1])
        type(mock_dat.return_value).label = PropertyMock(side_effect=[1, 0])

        x, l = r.read_values_at(self.path_lmdb, 0)
        for ch in range(3):
            for row in range(4):
                for col in range(2):
                    assert_equal(x[ch, row, col], self.img1_data[row, col, ch])
        assert_equal(l, 1, "Unexpected 1st label")

        x, l = r.read_values_at(self.path_lmdb, 1)
        img2_data = self.img1_data[2:, 1:, :]
        for ch in range(3):
            for row in range(2):
                for col in range(1):
                    assert_equal(x[ch, row, col], img2_data[row, col, ch])
        assert_equal(l, 0, "Unexpected 2nd label")
        
    @patch('nideep.iow.read_lmdb.caffe.proto.caffe_pb2.Datum')
    def test_read_values_at_key_string(self, mock_dat):

        # mocks
        dstr = ['\x01\x04\x07\n\r\x10\x13\x16\x02\x05\x08\x0b\x0e\x11\x14\x17\x03\x06\t\x0c\x0f\x12\x15\x18',
                '\x10\x16\x11\x17\x12\x18']

        mock_dat.return_value.ParseFromString.return_value = ""
        type(mock_dat.return_value).data = PropertyMock(side_effect=dstr)
        type(mock_dat.return_value).channels = PropertyMock(side_effect=[3, 3])
        type(mock_dat.return_value).height = PropertyMock(side_effect=[4, 2])
        type(mock_dat.return_value).width = PropertyMock(side_effect=[2, 1])
        type(mock_dat.return_value).label = PropertyMock(side_effect=[1, 0])

        x, l = r.read_values_at(self.path_lmdb, "0")
        for ch in range(3):
            for row in range(4):
                for col in range(2):
                    assert_equal(x[ch, row, col], self.img1_data[row, col, ch])
        assert_equal(l, 1, "Unexpected 1st label")

        x, l = r.read_values_at(self.path_lmdb, "1")
        img2_data = self.img1_data[2:, 1:, :]
        for ch in range(3):
            for row in range(2):
                for col in range(1):
                    assert_equal(x[ch, row, col], img2_data[row, col, ch])
        assert_equal(l, 0, "Unexpected 2nd label")
        
    @raises(TypeError)
    def test_read_values_at_out_of_bounds(self):

        r.read_values_at(self.path_lmdb, 2)

class TestReadArraysLMDB:

    def setup(self):

        self.dir_tmp = tempfile.mkdtemp()

        self.img1_data = np.array([[[ 1, 2, 3],
                                    [ 4, 5, 6]
                                    ],
                                   [[ 7, 8, 9],
                                    [10, 11, 12]
                                    ],
                                   [[13, 14, 15],
                                    [16, 17, 18],
                                    ],
                                   [[19, 20, 21],
                                    [22, 23, 24]
                                    ]
                                   ])

        img_data_str = ['\x08\x04\x10\x02\x18\x03(\x005\x00\x00\x80?5\x00\x00\x00@5\x00\x00@@5\x00\x00\x80@5\x00\x00\xa0@5\x00\x00\xc0@5\x00\x00\xe0@5\x00\x00\x00A5\x00\x00\x10A5\x00\x00 A5\x00\x000A5\x00\x00@A5\x00\x00PA5\x00\x00`A5\x00\x00pA5\x00\x00\x80A5\x00\x00\x88A5\x00\x00\x90A5\x00\x00\x98A5\x00\x00\xa0A5\x00\x00\xa8A5\x00\x00\xb0A5\x00\x00\xb8A5\x00\x00\xc0A',
                        '\x08\x02\x10\x01\x18\x03(\x005\x00\x00\x80A5\x00\x00\x88A5\x00\x00\x90A5\x00\x00\xb0A5\x00\x00\xb8A5\x00\x00\xc0A']

        # write fake data to lmdb
        self.path_lmdb = os.path.join(self.dir_tmp, 'imgs_lmdb')
        db = lmdb.open(self.path_lmdb, map_size=int(1e12))
        with db.begin(write=True) as in_txn:

            for idx, data_str in enumerate(img_data_str):
                in_txn.put('{:0>10d}'.format(idx), data_str)
        db.close()

    def teardown(self):

        shutil.rmtree(self.dir_tmp)

    @patch('nideep.iow.read_lmdb.caffe.proto.caffe_pb2.Datum')
    def test_read_values(self, mock_dat):

        img2_data = self.img1_data[2:, 1:, :]

        # mocks
        df = [self.img1_data.astype(float).flatten().ravel().tolist(),
              img2_data.astype(float).flatten().ravel().tolist()]

        mock_dat.return_value.ParseFromString.return_value = ""
        type(mock_dat.return_value).data = PropertyMock(return_value='')
        type(mock_dat.return_value).float_data = PropertyMock(side_effect=df)
        type(mock_dat.return_value).channels = PropertyMock(side_effect=[4, 2])
        type(mock_dat.return_value).height = PropertyMock(side_effect=[2, 1])
        type(mock_dat.return_value).width = PropertyMock(side_effect=[3, 3])

        v = r.read_values(self.path_lmdb)

        assert_equal(len(v), 2, "Unexpected no. of elements.")

        x, _ = v[0]
        for ch in range(4):
            for row in range(2):
                for col in range(3):
                    assert_equal(x[ch, row, col], self.img1_data[ch, row, col])

        x, _ = v[1]

        for ch in range(2):
            for row in range(1):
                for col in range(3):
                    assert_equal(x[ch, row, col], img2_data[ch, row, col])

    @patch('nideep.iow.read_lmdb.caffe.proto.caffe_pb2.Datum')
    def test_read_values_float(self, mock_dat):

        img2_data = self.img1_data[2:, 1:, :]

        # mocks
        df = [self.img1_data.astype(float).flatten().ravel().tolist(),
              img2_data.astype(float).flatten().ravel().tolist()]

        mock_dat.return_value.ParseFromString.return_value = ""
        type(mock_dat.return_value).data = PropertyMock(return_value='')
        type(mock_dat.return_value).float_data = PropertyMock(side_effect=df)
        type(mock_dat.return_value).channels = PropertyMock(side_effect=[4, 2])
        type(mock_dat.return_value).height = PropertyMock(side_effect=[2, 1])
        type(mock_dat.return_value).width = PropertyMock(side_effect=[3, 3])

        v = r.read_values(self.path_lmdb, float)

        assert_equal(len(v), 2, "Unexpected no. of elements.")

        x, _ = v[0]
        for ch in range(4):
            for row in range(2):
                for col in range(3):
                    assert_equal(x[ch, row, col], self.img1_data[ch, row, col])

        x, _ = v[1]

        for ch in range(2):
            for row in range(1):
                for col in range(3):
                    assert_equal(x[ch, row, col], img2_data[ch, row, col])

    # to generate mock data
#     def test_gen_float_lmdb(self):
#
#         import to_lmdb
#         self.img1 = np.array([[[ 1,  2,  3],
#                                [ 4,  5,  6]
#                                ],
#                               [[ 7,  8,  9],
#                                [10, 11, 12]
#                                ],
#                               [[13, 14, 15],
#                                [16, 17, 18],
#                                ],
#                               [[19, 20, 21],
#                                [22, 23, 24]
#                                ]
#                               ])
#
#         self.img1 = self.img1.astype(np.float)
#         print self.img1.shape
#
#         print "-------"
#         path_lmdb = os.path.join(self.dir_tmp, 'imgsX1_lmdb')
#         to_lmdb.arrays_to_lmdb([self.img1, self.img1[2:,1:,:]], path_lmdb)
#
#         v = r.read_values(path_lmdb)
#         print v
#         print "-------"

class TestNumEntriesLMDB:

    def setup(self):

        self.dir_tmp = tempfile.mkdtemp()

        self.img1_data = np.array([[[ 1, 2, 3],
                                    [ 4, 5, 6]
                                    ],
                                   [[ 7, 8, 9],
                                    [10, 11, 12]
                                    ],
                                   [[13, 14, 15],
                                    [16, 17, 18],
                                    ],
                                   [[19, 20, 21],
                                    [22, 23, 24]
                                    ]
                                   ])

        img_data_str = ['\x08\x03\x10\x04\x18\x02"\x18\x01\x04\x07\n\r\x10\x13\x16\x02\x05\x08\x0b\x0e\x11\x14\x17\x03\x06\t\x0c\x0f\x12\x15\x18(\x01',
                        '\x08\x03\x10\x02\x18\x01"\x06\x10\x16\x11\x17\x12\x18(\x00']

        # write fake data to lmdb
        self.path_lmdb = os.path.join(self.dir_tmp, 'imgs_lmdb')
        db = lmdb.open(self.path_lmdb, map_size=int(1e12))
        with db.begin(write=True) as in_txn:

            for idx, data_str in enumerate(img_data_str):
                in_txn.put('{:0>10d}'.format(idx), data_str)
        db.close()

    def teardown(self):

        shutil.rmtree(self.dir_tmp)

    def test_num_entries(self):

        assert_equal(2, r.num_entries(self.path_lmdb))

    def test_num_entries_empty(self):

        path_lmdb_empty = os.path.join(self.dir_tmp, 'empty_lmdb')
        db = lmdb.open(path_lmdb_empty, map_size=int(1e12))
        with db.begin(write=True) as _:
            _
        db.close()

        assert_equal(0, r.num_entries(path_lmdb_empty))

    def test_num_entries_does_not_exist(self):

        path_lmdb = os.path.join(self.dir_tmp, 'test_num_entries_does_not_exist_lmdb')
        assert_false(os.path.exists(path_lmdb))
        assert_raises(lmdb.Error, r.num_entries, path_lmdb)

class TestNumEntriesNumericOrderedLMDB:

    def setup(self):

        self.dir_tmp = tempfile.mkdtemp()

        self.img1_data = np.array([[[ 1, 2, 3],
                                    [ 4, 5, 6]
                                    ],
                                   [[ 7, 8, 9],
                                    [10, 11, 12]
                                    ],
                                   [[13, 14, 15],
                                    [16, 17, 18],
                                    ],
                                   [[19, 20, 21],
                                    [22, 23, 24]
                                    ]
                                   ])

        img_data_str = ['\x08\x03\x10\x04\x18\x02"\x18\x01\x04\x07\n\r\x10\x13\x16\x02\x05\x08\x0b\x0e\x11\x14\x17\x03\x06\t\x0c\x0f\x12\x15\x18(\x01',
                        '\x08\x03\x10\x02\x18\x01"\x06\x10\x16\x11\x17\x12\x18(\x00']

        # write fake data to lmdb
        self.path_lmdb_num_ord = os.path.join(self.dir_tmp, 'imgs_num_ord_lmdb')
        db = lmdb.open(self.path_lmdb_num_ord, map_size=int(1e12))
        with db.begin(write=True) as in_txn:

            for idx, data_str in enumerate(img_data_str):
                in_txn.put('{:0>10d}'.format(idx), data_str)
        db.close()

        self.path_lmdb_rand_ord = os.path.join(self.dir_tmp, 'imgs_rand_ord_lmdb')
        db = lmdb.open(self.path_lmdb_rand_ord, map_size=int(1e12))
        with db.begin(write=True) as in_txn:

            for data_str in img_data_str:
                in_txn.put('{:0>10d}'.format(np.random.randint(10, 1000)), data_str)
        db.close()

        self.path_lmdb_non_num = os.path.join(self.dir_tmp, 'imgs_non_num_lmdb')
        db = lmdb.open(self.path_lmdb_non_num, map_size=int(1e12))
        with db.begin(write=True) as in_txn:

            for data_str in img_data_str:
                in_txn.put('key' + data_str, data_str)
        db.close()

        assert_not_equal(self.path_lmdb_num_ord, self.path_lmdb_rand_ord)
        assert_not_equal(self.path_lmdb_num_ord, self.path_lmdb_non_num)
        assert_not_equal(self.path_lmdb_rand_ord, self.path_lmdb_non_num)

    def teardown(self):

        shutil.rmtree(self.dir_tmp)

    def test_num_entries_num_ord(self):

        assert_equal(2, r.num_entries(self.path_lmdb_num_ord))

    def test_num_entries_rand_ord(self):

        assert_equal(2, r.num_entries(self.path_lmdb_rand_ord))

    def test_num_entries_non_num(self):

        assert_equal(2, r.num_entries(self.path_lmdb_non_num))

