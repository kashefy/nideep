import os
import numpy as np
import cv2
import argparse
import shutil
import pandas as pd
import logging
from sklearn.model_selection import train_test_split


class WashingtonRGBD(object):
    """
    
    Data Wrapper class for WashingtonRGBD dataset
    Attributes
    -----------
    root_dir: root directory until the rgbd-dataset folder. For example: /mnt/raid/data/ni/dnn/pduy/rgbd-dataset
    csv_default: the default directory for loading/saving the csv description of the dataset
    csv_interpolated_default: the default directory for loading/saving the pose-interpolated csv description of the 
    dataset.
    
    """

    def __init__(self, root_dir='', csv_default='', csv_interpolated_default=''):
        self.logger = logging.getLogger(__name__)
        self.root_dir = root_dir
        self.csv_default = csv_default
        self.csv_interpolated_default = csv_interpolated_default

    # Load the dataset metadata to a Pandas dataframe and save the result to a csv file
    # if it does not exists
    # otherwise read the csv
    # The missing pose values will be saved as -1
    def load_metadata(self):
        if os.path.isfile(self.csv_default):
            self.logger.info('reading from ' + self.csv_default)
            return pd.read_csv(self.csv_default)

        file_list = os.walk(self.root_dir)

        data = []

        for current_root, _, files in file_list:
            # For the time being, we do not work on the mask, location
            # For the pose, it should be attached to the corresponding data entry, not as a separate entry
            files = [f for f in files if 'mask' not in f and 'loc' not in f and 'pose' not in f]

            for f in files:
                self.logger.info("processing " + f)

                pose_value = -1

                name_components = f.split('_')

                # The category name can be 1 word or 2 words, such as 'apple' or 'cell_phone'
                # So, when splitting the file name by '_', there can be 5 or 6 components
                # That's why I read the name backward to make sure I get the proper data pieces
                # reversed_name_components = np.flip(name_components, axis=0)

                if len(name_components) < 5:
                    continue

                n_components = len(name_components)
                if n_components > 5:    # if n_components > 5, it means the category name has more than 1 word
                    category = '_'.join(name_components[0: n_components - 4])
                else:
                    category = name_components[0]

                instance_number = name_components[-4]
                video_no = name_components[-3]
                frame_no = name_components[-2]
                data_type = name_components[-1].split('.')[0]

                name_components[n_components - 1] = 'pose.txt'
                pose_file_name = '_'.join(name_components)

                try:
                    with open(os.path.join(current_root, pose_file_name), 'r') as pose_file:
                        pose_value = pose_file.readline()
                        self.logger.info("pose value = " + str(pose_value))
                except IOError:
                    self.logger.info("No pose value for this instance!")

                data.append({'location': os.path.join(current_root, f),
                             'category': category,
                             'instance_number': int(instance_number),
                             'video_no': int(video_no),
                             'frame_no': int(frame_no),
                             'pose': float(pose_value),
                             'data_type': data_type})

            data_frame = pd.DataFrame(data) \
                .sort_values(['data_type', 'category', 'instance_number', 'video_no', 'frame_no'])

            self.logger.info("csv saved to file: " + self.csv_default + '.csv')
            data_frame.to_csv(self.csv_default, index=False)

        return data_frame

    # Interpolate the missing pose values (saved as -1 by the load_metadata() method)
    def interpolate_poses(self, data_frame):
        if os.path.isfile(self.csv_interpolated_default):
            self.logger.info('reading from ' + self.csv_interpolated_default)
            return pd.read_csv(self.csv_interpolated_default)

        self.logger.info('Interpolating ...')

        sorted_df = data_frame.sort_values(['data_type', 'category', 'instance_number', 'video_no', 'frame_no'])
        poses = np.array(sorted_df['pose'])

        current_video = -1

        for i in range(0, len(poses)):
            if (sorted_df['video_no'][i] != current_video) and (poses[i] == 0):
                unit_diff_angle = poses[i + 5] / 5

            if poses[i] == -1:
                poses[i] = poses[i - 1] + unit_diff_angle
                if poses[i] > 360:
                    poses[i] = poses[i] - 360

        sorted_df['pose'] = poses
        sorted_df.to_csv(self.csv_interpolated_default, index=False)

        self.logger.info('Interpolation finished!')

        return sorted_df

    def extract_rgb_only(self, output_path):
        data_frame = self.load_metadata()
        rgb_files = data_frame[data_frame['data_type'] == 'crop']['location']

        for f in rgb_files:
            shutil.copy(os.path.join(self.root_dir, f), output_path)

    # Combine an rgb image with a rotated image of the same object horizontally into 1 image,
    # together with a train-test-split for doing a hold-out validation.
    # Only one elevation video_no is taken, the other elevations are ignored
    # Left:RGB, (Middle: Depth Map),  Right: Rotation
    def combine_viewpoints(self, angle, video_no, should_include_depth, output_path):

        def join_rgb_with_rotation(data_frame, output_path):
            data_frame = data_frame[data_frame['data_type'] == 'crop']
            for i in range(len(data_frame.index)):
                current_original_file_df = data_frame.iloc[[i]]

                # Filtering out the rotation candidates,
                # most of the things should be the same, except for frame_no,
                # and the 2 poses should differentiate by the provided angle with an error bound of +-1
                rotation_candidates = data_frame[(data_frame['category'] == current_original_file_df['category'].values[0])
                                                 & (data_frame['instance_number'] == current_original_file_df['instance_number'].values[0])
                                                 & (data_frame['video_no'] == current_original_file_df['video_no'].values[0])
                                                 & (data_frame['pose'] <= current_original_file_df['pose'].values[0] + angle + 1)
                                                 & (data_frame['pose'] >= current_original_file_df['pose'].values[0] + angle - 1)]

                for j in range(len(rotation_candidates.index)):
                    current_rotated_file_df = rotation_candidates.iloc[[j]]

                    left_location = current_original_file_df['location'].values[0]
                    right_location = current_rotated_file_df['location'].values[0]

                    left_img_name = left_location.split('/')[len(left_location.split('/')) - 1]
                    right_img_name = right_location.split('/')[len(right_location.split('/')) - 1]

                    self.logger.info("merging " + left_img_name + " and " + right_img_name)

                    left_img = cv2.imread(left_location)
                    right_img = cv2.imread(right_location)

                    smaller_height = min(len(left_img), len(right_img))
                    smaller_width = min(len(left_img[0]), len(right_img[0]))

                    left_img = cv2.resize(left_img, (smaller_width, smaller_height))
                    right_img = cv2.resize(right_img, (smaller_width, smaller_height))

                    img = np.concatenate((left_img, right_img), axis=1)
                    cv2.imwrite(os.path.join(output_path,
                                             '_'.join([os.path.splitext(left_img_name)[0],
                                                       right_img_name])), img)

        def join_rgb_depth_rotation(data_frame, output_path):
            original_df = data_frame[data_frame['data_type'] == 'crop']
            for i in range(len(original_df.index)):
                current_original_file_df = original_df.iloc[[i]]

                rotation_candidates = original_df[(original_df['category'] == current_original_file_df['category'].values[0])
                                                  & (original_df['instance_number'] == current_original_file_df['instance_number'].values[0])
                                                  & (original_df['video_no'] == current_original_file_df['video_no'].values[0])
                                                  & (original_df['pose'] <= current_original_file_df['pose'].values[0] + angle + 1)
                                                  & (original_df['pose'] >= current_original_file_df['pose'].values[0] + angle - 1)]

                depth_candidates = data_frame[(data_frame['category'] == current_original_file_df['category'].values[0])
                                              & (data_frame['instance_number'] == int(current_original_file_df['instance_number'].values[0]))
                                              & (data_frame['video_no'] == int(current_original_file_df['video_no'].values[0]))
                                              & (data_frame['frame_no'] == int(current_original_file_df['frame_no'].values[0]))
                                              & (data_frame['data_type'] == 'depthcrop')]

                for j in range(len(rotation_candidates.index)):
                    if len(depth_candidates.index) == 0:
                        continue

                    current_rotated_file_df = rotation_candidates.iloc[[j]]

                    left_location = current_original_file_df['location'].values[0]
                    middle_location = depth_candidates['location'].values[0]
                    right_location = current_rotated_file_df['location'].values[0]

                    left_img_name = os.path.split(left_location)[1]
                    middle_img_name = os.path.split(middle_location)[1]
                    right_img_name = os.path.split(right_location)[1]

                    self.logger.info("merging " + left_img_name + " and " + middle_img_name + " and " + right_img_name)

                    left_img = cv2.imread(left_location)
                    middle_img = cv2.imread(middle_location)
                    right_img = cv2.imread(right_location)

                    smaller_height = min(len(left_img), len(middle_img), len(right_img))
                    smaller_width = min(len(left_img[0]), len(middle_img[0]), len(right_img[0]))

                    left_img = cv2.resize(left_img, (smaller_width, smaller_height))
                    middle_img = cv2.resize(middle_img, (smaller_width, smaller_height))
                    right_img = cv2.resize(right_img, (smaller_width, smaller_height))

                    img = np.concatenate((left_img, middle_img, right_img), axis=1)
                    cv2.imwrite(os.path.join(output_path,
                                             '_'.join([os.path.splitext(left_img_name)[0],
                                                       os.path.splitext(middle_img_name)[0],
                                                       right_img_name])), img)

        def join(data_frame, output_path, should_include_depth):
            if output_path != '' and not os.path.isdir(output_path):
                os.makedirs(output_path)

            if should_include_depth:
                join_rgb_depth_rotation(data_frame, output_path)
            else:
                join_rgb_with_rotation(data_frame, output_path)

        data_frame = self.interpolate_poses(self.load_metadata())

        # Filter out the first elevator only
        data_frame = data_frame[data_frame['video_no'] == video_no]

        # train test split
        train, test = train_test_split(data_frame, test_size=0.2)

        # construct training and test sets, saving to disk
        join(train, os.path.join(output_path, 'train'), should_include_depth)
        join(test, os.path.join(output_path, 'test'), should_include_depth)


if __name__ == '__main__':
    ROOT_DEFAULT = '/mnt/raid/data/ni/dnn/pduy/rgbd-dataset'
    CSV_DEFAULT = '/mnt/raid/data/ni/dnn/pduy/rgbd-dataset/rgbd-dataset.csv'
    CSV_INTERPOLATED_DEFAULT = '/mnt/raid/data/ni/dnn/pduy/rgbd-dataset/rgbd-dataset-interpolated.csv'

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--rootdir", default=ROOT_DEFAULT)
    parser.add_argument("--csv_dir", default=CSV_DEFAULT)
    parser.add_argument("--csv_interpolated_dir", default=CSV_INTERPOLATED_DEFAULT)
    parser.add_argument("--processed_data_output", default='')
    parser.add_argument("--angle", default=10, type=int)
    parser.add_argument("--depth_included", default=False, type=bool)

    args = parser.parse_args()

    if args.processed_data_output != '' and not os.path.isdir(args.processed_data_output):
        os.makedirs(args.processed_data_output)

    # file_list = os.walk(args.rootdir)
    washington_dataset = WashingtonRGBD(args.rootdir, args.csv_dir, args.csv_interpolated_dir)
    washington_dataset.combine_viewpoints(angle=args.angle,
                                          video_no=1,
                                          should_include_depth=args.depth_included,
                                          output_path=args.processed_data_output)

