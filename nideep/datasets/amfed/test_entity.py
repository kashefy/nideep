'''
Created on Oct 28, 2015

@author: kashefy
'''
import os
import tempfile
import random
import shutil
import string
from nose.tools import assert_equals, assert_true, \
    assert_list_equal, assert_greater
from mock import patch, MagicMock
from entity import Entity
import numpy as np

def generate_random_entity_name():
    return ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(9))

def write_dummy_txt(p):
    with open(p, 'wb') as h:
        h.write("foo")
        
class TestEntity:
    
    NUM_FRAMES = 5
    
    @classmethod
    def setup_class(self):

        self.dir_tmp = tempfile.mkdtemp()
        self.entity_name = generate_random_entity_name()
        self.path_video = os.path.join(self.dir_tmp, self.entity_name + '.flv')
        self.path_au_labels = os.path.join(self.dir_tmp, self.entity_name + '-label.csv')
        self.path_landmarks = os.path.join(self.dir_tmp, self.entity_name + '-landmarks.txt')
        self.cache_dir = os.path.join(self.dir_tmp, 'cache/')
        
        # fake video files
        write_dummy_txt(os.path.join(self.path_video))
        write_dummy_txt(os.path.join(self.path_au_labels))
        write_dummy_txt(os.path.join(self.path_landmarks))
                        
    @classmethod
    def teardown_class(self):
        shutil.rmtree(self.dir_tmp)
        
    @patch('cv2.VideoCapture')
    def test_timesteps_sec(self, mock_cap):

        # set up fake videocapture
        node_instance = mock_cap.return_value
        pos_sec = np.sort(np.random.rand(TestEntity.NUM_FRAMES)*1000).tolist()
        node_instance.get = MagicMock(side_effect=pos_sec)
        node_instance.isOpened = \
            MagicMock(side_effect=[True]*(TestEntity.NUM_FRAMES) + [False])
        node_instance.grab.return_value = True
        
        e = Entity(self.path_video, self.path_au_labels, self.path_landmarks, self.cache_dir)
        t = e.timestamps_sec()
        assert_list_equal(t, [x/1000. for x in pos_sec])
        
    @patch('cv2.VideoCapture')
    def test_dense_labels(self, mock_cap):
        
        # set up fake videocapture
        node_instance = mock_cap.return_value
        pos_sec = [0.0, 0.2, 0.3, 0.6, 1.0, 1.2, 1.4]
        node_instance.get = MagicMock(side_effect=[x*1000. for x in pos_sec])
        node_instance.isOpened = \
            MagicMock(side_effect=[True]*len(pos_sec) + [False])
        node_instance.grab.return_value = True
        
        with open(self.path_au_labels, 'wb') as h:
            h.write("Time,Smile,AU04\n")
            h.write("%f,0.0,0.0\n" % pos_sec[0])
            h.write("%f,0.0,50.0\n" % pos_sec[1])
            h.write("%f,0.0,66.6\n" % pos_sec[3])
            h.write("%f,50.0,66.6\n" % pos_sec[4])
            h.write("%f,50.0,0.0\n" % pos_sec[5])
        
        e = Entity(self.path_video, self.path_au_labels, self.path_landmarks, self.cache_dir)
        labels, col_names = e.dense_labels()
        
        assert_list_equal(col_names, ["Smile", "AU04"], "unexpected column names")

        assert_equals(len(pos_sec), labels.shape[0], "mismatch between timestamps and dense label rows")
        assert_greater(labels.shape[0], 0, "Unexpected number of dense label rows")
        
        labels_expected = \
            [[  0.,   0. ], # 0.0, 
             [  0.,  50. ], # 0.2
             [  0.,  50. ], # 0.3
             [  0.,  66.6], # 0.6
             [ 50.,  66.6], # 1.0
             [ 50.,   0. ], # 1.2
             [ 50.,   0. ]] # 1.4
        for ti, row_expected, row in zip(pos_sec, labels_expected, labels):
            assert_list_equal(row.tolist(), row_expected, "unexpected dense labels for timestamp %f" % ti)
        
    @patch('cv2.VideoCapture')
    def test_dense_labels_all_zero(self, mock_cap):
        
        # set up fake videocapture
        node_instance = mock_cap.return_value
        pos_sec = np.sort(np.random.rand(TestEntity.NUM_FRAMES)*1000).tolist()
        node_instance.get = MagicMock(side_effect=pos_sec)
        node_instance.isOpened = \
            MagicMock(side_effect=[True]*(TestEntity.NUM_FRAMES) + [False])
        node_instance.grab.return_value = True
        
        with open(self.path_au_labels, 'wb') as h:
            h.write("Time,Smile,AU04\n")
            h.write("0.0,0.0,0.0\n")
            
        e = Entity(self.path_video, self.path_au_labels, self.path_landmarks, self.cache_dir)
        labels, col_names = e.dense_labels()
        
        assert_list_equal(col_names, ["Smile", "AU04"], "unexpected column names")

        assert_equals(len(pos_sec), labels.shape[0], "mismatch between timestamps and dense label rows")
        assert_greater(labels.shape[0], 0, "Unexpected number of dense label rows")
        
        for ti, row in zip(pos_sec, labels):
            assert_list_equal(row.tolist(), [0.0]*2, "unexpected dense labels for timestamp %f" % ti)
