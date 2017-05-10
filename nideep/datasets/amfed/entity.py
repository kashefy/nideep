import os
import logging
import numpy as np
import cv2
import pandas as pd
import cPickle as pickle
from preprocessing import crop_face
from PIL import Image


class Entity(object):
    def timestamps_sec(self):
        """ Get timestamp of each frame from video file """
        cap = cv2.VideoCapture(self.path_video)
        ret = True
        t = []
        while cap.isOpened() and ret:
            # Capture frame-by-frames
            t.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.)
            ret = cap.grab()
        # When everything done, release the capture
        cap.release()
        return t

    def frames(self, projection=[], valid_only=False):
        """
        Get a list of dicts corresponding to the decoded frames of the video. Each dict has the following keys:
        - 'features': if valid, the numpy array corresponding to the frame else None
        - 'labels': a numpy array corresponding to the labels
        - 'valid': whether the frame has labels and landmarks available, and is not a trackerfail
        - 'message': additional information. if invalid, a possible hint of the reason
        """

        labels_dict = self.dense_labels_dict()
        selected_labels = projection if projection else labels_dict.keys()
        vidcap = cv2.VideoCapture(self.path_video)
        success, image = vidcap.read()
        count = 0
        result = []

        while success:
            frame_file = self.__frame_filename(count)

            labels = [labels_dict[k][count] if k in labels_dict.keys() else None for k in selected_labels]
            binarized_labels = [1 if l >= 50 else 0 for l in labels]

            record = {'features': None, 'labels': np.array(binarized_labels), 'message': '', 'valid': True}

            if not labels:
                record['valid'] = False
                record['message'] = 'No labels available'
            elif None in labels:
                record['valid'] = False
                record['message'] = 'One or more selected columns are not available'
            else:
                if os.path.isfile(frame_file):
                    # only valid frames are dumped to disk
                    record['features'] = cv2.imread(frame_file)
                    record['message'] = 'Cached version read'
                else:
                    success, msg, value = self.preprocess_frame(image, count, labels, projection)
                    record['valid'] = success
                    record['message'] = msg
                    record['features'] = value

            if record['valid'] or not valid_only:
                result.append(record)
            success, image = vidcap.read()
            count += 1

        return result

    def preprocess_frame(self, frame, frame_number, labels, projection):
        frame_file = self.__frame_filename(frame_number)

        if frame_number < self.landmarks_df.shape[0]:
            left_eye = (
                self.landmarks_df.iloc[[frame_number]][0][frame_number],
                self.landmarks_df.iloc[[frame_number]][1][frame_number])
            right_eye = (
                self.landmarks_df.iloc[[frame_number]][2][frame_number],
                self.landmarks_df.iloc[[frame_number]][3][frame_number])
        else:
            return False, 'Frame number > number of landmark file rows', None

        if all(v == 0 for v in (left_eye + right_eye)):
            return False, 'Tracker fail: no landmarks available', None

        if labels[-1]:
            return False, 'Tracker fail: human labeled failure', None

        cropped = np.array(crop_face(Image.fromarray(frame), eye_left=left_eye, eye_right=right_eye,
                                     offset_pct=(0.25, 0.25), dest_sz=(64, 64)))

        # gray_image = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

        cv2.imwrite(frame_file, cropped)  # save frame as PNG file
        return True, 'Success', cv2.imread(frame_file)

    def dense_labels_dict(self):
        """ Load dense labels for each column in au labels file into dictionary"""
        labels, col_names = self.dense_labels()
        labels_dict = {}
        for idx, c in enumerate(col_names):
            labels_dict[c] = labels[:, idx]
        return labels_dict

    def dense_labels(self):
        """ Get dense labels for all columns in au labels """
        onsets = np.genfromtxt(self.path_au_labels, dtype=float, delimiter=',', names=True)
        t_dense = self.timestamps_sec()
        cols = list(onsets.dtype.names[1:])  # omit Time from header
        labels = np.zeros((len(t_dense), len(cols)), dtype='float')
        if onsets.shape and len(onsets) > 1:
            for cur, next in zip(onsets[:-1], onsets[1:]):
                t_on = cur[0]
                t_off = next[0]
                valid_rows = np.logical_and(t_dense >= t_on, t_dense < t_off)
                labels[valid_rows, :] = list(cur)[1:]
            labels[t_dense >= t_off, :] = list(next)[1:]  # fill remaining rows
        return labels, cols

    def __labels_filename(self, frame_number):
        video_name = self.path_video.split('/')[-1][:-4]
        return self.cache_dir + video_name + '-labels%d.p' % frame_number

    def __frame_filename(self, frame_number):
        video_name = self.path_video.split('/')[-1][:-4]
        return self.cache_dir + video_name + '-frame%d.png' % frame_number

    def __init__(self, path_video, path_au_labels, path_landmarks, cache_dir):
        '''
        path_video -- path to AVI/FLV file
        path_au_labels -- path to csv file with timestamped au labels
        path_landmarks -- path to landmarks text file with row per timestamps
        cache_dir -- path to dump preprocessed frames
        '''
        self.cache_dir = cache_dir
        self.logger = logging.getLogger(__name__)
        self.path_video = path_video
        if not os.path.isfile(self.path_video):
            self.logger.warning("Video file does not exist (%s)" % self.path_video)
        self.path_au_labels = path_au_labels
        if not os.path.isfile(self.path_au_labels):
            self.logger.warning("AU Labels file does not exist (%s)" % self.path_au_labels)
        self.path_landmarks = path_landmarks
        if not os.path.isfile(self.path_landmarks):
            self.logger.warning("Landmarks file does not exist (%s)" % self.path_landmarks)
        self.landmarks_df = pd.read_csv(self.path_landmarks, header=None)
