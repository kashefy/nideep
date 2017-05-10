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
    assert_list_equal
from amfed import AMFED


def generate_random_entity_name():
    return ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(9))


def write_dummy_txt(p):
    with open(p, 'wb') as h:
        h.write("foo")


class TestAMFED:
    @classmethod
    def setup_class(self):

        self.dir_tmp = tempfile.mkdtemp()
        self.dir_videos = os.path.join(self.dir_tmp, AMFED.DIRNAME_FLV)
        self.dir_au_labels = os.path.join(self.dir_tmp, AMFED.DIRNAME_AU_LABELS)
        self.dir_landmarks = os.path.join(self.dir_tmp, AMFED.DIRNAME_LANDMARKS)
        os.mkdir(self.dir_videos)
        os.mkdir(self.dir_au_labels)
        os.mkdir(self.dir_landmarks)

        # generate fake entity names
        self.entity_names = [generate_random_entity_name() for _ in xrange(7)]

        # fake video files
        write_dummy_txt(os.path.join(self.dir_videos, self.entity_names[0] + '.flv'))
        write_dummy_txt(os.path.join(self.dir_videos, self.entity_names[2] + '.flv'))
        write_dummy_txt(os.path.join(self.dir_videos, self.entity_names[3] + '.flv'))
        # fake au labels files
        write_dummy_txt(os.path.join(self.dir_au_labels, self.entity_names[1] + AMFED.SUFFIX_AU_LABELS + '.csv'))
        write_dummy_txt(os.path.join(self.dir_au_labels, self.entity_names[2] + AMFED.SUFFIX_AU_LABELS + '.csv'))
        write_dummy_txt(os.path.join(self.dir_au_labels, self.entity_names[3] + AMFED.SUFFIX_AU_LABELS + '.csv'))
        write_dummy_txt(os.path.join(self.dir_au_labels, self.entity_names[4] + AMFED.SUFFIX_AU_LABELS + '.csv'))
        # fake landmarks files
        write_dummy_txt(os.path.join(self.dir_landmarks, self.entity_names[2] + AMFED.SUFFIX_LANDMARKS + '.txt'))
        write_dummy_txt(os.path.join(self.dir_landmarks, self.entity_names[3] + AMFED.SUFFIX_LANDMARKS + '.txt'))
        write_dummy_txt(os.path.join(self.dir_landmarks, self.entity_names[5] + AMFED.SUFFIX_LANDMARKS + '.txt'))

    @classmethod
    def teardown_class(self):

        shutil.rmtree(self.dir_tmp)

    def test_intersection(self):
        amfed = AMFED(self.dir_tmp)
        found_entities = amfed.find_data_intersection()

        assert_equals(len(found_entities), 2)

        idxs_found = []
        for _, e in found_entities.iterrows():
            v = e.video
            a = e.au_label
            l = e.landmarks
            assert_true(os.path.isfile(v))
            assert_true(os.path.isfile(a))
            assert_true(os.path.isfile(l))
            found = False
            idx = 0
            while idx < len(self.entity_names) and not found:
                e = self.entity_names[idx]
                if os.path.splitext(os.path.basename(v))[0] == e:
                    assert_equals(v, os.path.join(self.dir_videos, e + '.flv'))
                    assert_equals(a, os.path.join(self.dir_au_labels, e + AMFED.SUFFIX_AU_LABELS + '.csv'))
                    assert_equals(l, os.path.join(self.dir_landmarks, e + AMFED.SUFFIX_LANDMARKS + '.txt'))
                    idxs_found.append(idx)
                idx += 1
        idxs_found.sort()
        assert_list_equal(idxs_found, [2, 3])
