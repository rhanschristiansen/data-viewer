import pykitti
import pandas as pd
import src.util.calibration_kitti as cal
import cv2
import numpy as np
from src.detection.car_detector_tf_v2 import CarDetectorTFV2
from src.association.costs import Costs


class Preprocess_Kitti():
    def __init__(self):
        self.detector = CarDetectorTFV2()

    def get_video_detections(self, run_date, run_number, lidar_fov = [0,0,1242,375], det_classes = [], eliminate_redundant = False, confidence_threshold = 0.7):
        kitti_dataset = pykitti.raw(base_path='../data/', date=run_date, drive=run_number)

        lidar_left = lidar_fov[0]
        lidar_top = lidar_fov[1]
        lidar_right = lidar_fov[2]
        lidar_bottom = lidar_fov[3]
        det_classes = det_classes
        det_values_filtered = []
        det_values_unfiltered = []

        num_files = len(kitti_dataset.cam2_files)
        c = Costs(data_type='kitti')

        for frame_num, frame_filename in enumerate(kitti_dataset.cam2_files):
            print('Processing Frame: {0:0.0f} of {1:0.0f}'.format(frame_num, num_files))

            frame = cv2.imread(frame_filename)

            # get the video detections
            bbs, class_names, confidences = self.detector.detect(img=frame, return_class_scores=True)
            for i, bb in enumerate(bbs):
                # ensure bb is inside the window
                bbs[i][0] = max(bb[0],0)
                bbs[i][1] = max(bb[1],0)
                bbs[i][2] = min(bb[2],cal.cal['X_RESOLUTION'])
                bbs[i][3] = min(bb[3],cal.cal['Y_RESOLUTION'])
            n = len(class_names)

            for i, (bbox, class_name, confidence) in enumerate(zip(bbs, class_names, confidences)):
                det_values_unfiltered.append([frame_num, i, class_name, confidence, bbox[0], bbox[1], bbox[2], bbox[3]])

            eliminate_flag = np.zeros(n,np.int)
            for i, (bbox, class_name, confidence) in enumerate(zip(bbs, class_names, confidences)):
                # filter out bounding boxes that do not intersect with the lidar zone and are not vehicles
                if len(det_classes) > 0:
                    if class_name not in det_classes or confidence <= confidence_threshold:
                        eliminate_flag[i] = 1
                else:
                    if confidence <= confidence_threshold:
                        eliminate_flag[i] = 1

                if (bbox[1] > lidar_bottom or bbox[3] < lidar_top or bbox[2] < lidar_left or bbox[0] > lidar_right):
                    eliminate_flag[i] = 1
            #remove the items from bbs, class_names and confidences
            new_bbs = []; new_class_names = []; new_confidences = [];
            for ii in range(n):
                if eliminate_flag[ii] == 0:
                    new_bbs.append(bbs[ii])
                    new_class_names.append(class_names[ii])
                    new_confidences.append(confidences[ii])
            bbs = new_bbs
            class_names = new_class_names
            confidences = new_confidences

            n = len(class_names)
            eliminate_flag = np.zeros(n,np.int)

            #eliminate redundant detections - iou greater that 0.7 and different class_name
            if eliminate_redundant:
                for ii in range(n):
                    for jj in range(n):
                        if ii < jj and c._iou(bbs[ii],bbs[jj]) >= 0.7 and class_names[ii] != class_names[jj]:
                            eliminate_flag[jj] = 1

                #remove the redundant items from bbs, class_names and confidences
                new_bbs = []; new_class_names = []; new_confidences = [];
                for ii in range(n):
                    if eliminate_flag[ii] == 0:
                        new_bbs.append(bbs[ii])
                        new_class_names.append(class_names[ii])
                        new_confidences.append(confidences[ii])
                bbs = new_bbs
                class_names = new_class_names
                confidences = new_confidences
            # only append what's left
            for i, (bbox, class_name, confidence) in enumerate(zip(bbs, class_names, confidences)):
                det_values_filtered.append([frame_num, i, class_name, confidence, bbox[0], bbox[1], bbox[2], bbox[3]])

        columns = ['frame', 'detection_index', 'detection_class', 'detection_confidence', 'x1', 'y1', 'x2', 'y2']
        det_filtered_df = pd.DataFrame(det_values_filtered, columns=columns)
        det_unfiltered_df = pd.DataFrame(det_values_unfiltered, columns=columns)
        return det_filtered_df, det_unfiltered_df

    def get_lidar_detections(self, run_date, run_number, lidar_fov = [0,0,1242,375], lidar_range = [], gt_classes = [], det_classes = []):
        self.data_dir = '../data'
        self.run_date = run_date
        self.run_number = run_number
        self.kitti_dataset = pykitti.raw(base_path='../data/', date=run_date, drive=run_number)
        self.lidar_left = lidar_fov[0]
        self.lidar_top = lidar_fov[1]
        self.lidar_right = lidar_fov[2]
        self.lidar_bottom = lidar_fov[3]
        self.lidar_dist_min = lidar_range[0]
        self.lidar_dist_max = lidar_range[1]
        self.gt_classes = gt_classes
        self.det_classes = det_classes
        self.m16 = None
        self.gt_df = None
        self.det_df = None
        self.detector = CarDetectorTFV2()
        self.confidence_threshold = 0.8
        self.det_values = []
        # read the lidar data
        self.m16 = pd.read_csv('{}/{}/{}_filtered.csv'.format(self.data_dir, self.run_date, self.run_number), skiprows=2)
        return self.m16


    def get_ground_truth(self, run_date, run_number, lidar_fov = [0,0,1242,375], lidar_range = [], gt_classes = [], det_classes = []):
        self.data_dir = '../data'
        self.run_date = run_date
        self.run_number = run_number
        self.kitti_dataset = pykitti.raw(base_path='../data/', date=run_date, drive=run_number)
        self.lidar_left = lidar_fov[0]
        self.lidar_top = lidar_fov[1]
        self.lidar_right = lidar_fov[2]
        self.lidar_bottom = lidar_fov[3]
        self.lidar_dist_min = lidar_range[0]
        self.lidar_dist_max = lidar_range[1]
        self.gt_classes = gt_classes
        self.det_classes = det_classes
        self.m16 = None
        self.gt_df = None
        self.det_df = None
        self.detector = CarDetectorTFV2()
        self.confidence_threshold = 0.8
        self.det_values = []
        # read the ground truth from the tracklets data
        self.gt_df = pd.read_csv('{}/{}'.format(self.data_dir, '2011_09_26_drive_' + self.run_number + '_sync_converted-tracklets.csv'))
        self.gt_df['dist'] = self.gt_df['dist'] * cal.cal['M_TO_FT']
        # remove all objects that are not in the object_classes list (if list is empty skip this step
        if len(self.gt_classes) > 0:
            self.gt_df = self.gt_df[(self.gt_df['label'].isin(self.gt_classes))]
        # remove all objects outside the range of the lidar detector
        self.gt_df = self.gt_df[(self.gt_df['dist'] >= self.lidar_dist_min) & (self.gt_df['dist'] <= self.lidar_dist_max)]
        # remove all objects outside of the lidar fov
        self.gt_df = self.gt_df[self.gt_df['x1'] <= self.lidar_right]
        self.gt_df = self.gt_df[self.gt_df['x2'] >= self.lidar_left]
        self.gt_df = self.gt_df[self.gt_df['y1'] <= self.lidar_bottom]
        self.gt_df = self.gt_df[self.gt_df['y2'] >= self.lidar_top]

        return self.gt_df


if __name__ == "__main__":
    lidar_fov = [cal.SEG_TO_PIXEL_LEFT[0], cal.SEG_TO_PIXEL_TOP, cal.SEG_TO_PIXEL_RIGHT[15], cal.SEG_TO_PIXEL_BOTTOM ]
    lidar_range = [30, 140]
    gt_classes = ['Car', 'Truck', 'Van']
    det_classes = ['car', 'truck', 'bus']
    run_date = '2011_09_26'

    pre_kitti = Preprocess_Kitti()

    run_number = '0001'
    det_filtered_df, det_unfiltered_df = pre_kitti.get_video_detections(run_date, run_number=run_number, lidar_fov = lidar_fov, det_classes = det_classes, eliminate_redundant = True, confidence_threshold = 0.8)
    det_filename = '../data/{}_drive_{}_detections_filtered.csv'.format(run_date, run_number)
    det_filtered_df.to_csv(det_filename)
    det_filename = '../data/{}_drive_{}_detections_unfiltered.csv'.format(run_date, run_number)
    det_unfiltered_df.to_csv(det_filename)

    run_number = '0002'
    det_filtered_df, det_unfiltered_df = pre_kitti.get_video_detections(run_date, run_number=run_number, lidar_fov = lidar_fov, det_classes = det_classes, eliminate_redundant = True, confidence_threshold = 0.8)
    det_filename = '../data/{}_drive_{}_detections_filtered.csv'.format(run_date, run_number)
    det_filtered_df.to_csv(det_filename)
    det_filename = '../data/{}_drive_{}_detections_unfiltered.csv'.format(run_date, run_number)
    det_unfiltered_df.to_csv(det_filename)

    run_number = '0005'
    det_filtered_df, det_unfiltered_df = pre_kitti.get_video_detections(run_date, run_number=run_number, lidar_fov = lidar_fov, det_classes = det_classes, eliminate_redundant = True, confidence_threshold = 0.8)
    det_filename = '../data/{}_drive_{}_detections_filtered.csv'.format(run_date, run_number)
    det_filtered_df.to_csv(det_filename)
    det_filename = '../data/{}_drive_{}_detections_unfiltered.csv'.format(run_date, run_number)
    det_unfiltered_df.to_csv(det_filename)

    run_number = '0009'
    det_filtered_df, det_unfiltered_df = pre_kitti.get_video_detections(run_date, run_number=run_number, lidar_fov = lidar_fov, det_classes = det_classes, eliminate_redundant = True, confidence_threshold = 0.8)
    det_filename = '../data/{}_drive_{}_detections_filtered.csv'.format(run_date, run_number)
    det_filtered_df.to_csv(det_filename)
    det_filename = '../data/{}_drive_{}_detections_unfiltered.csv'.format(run_date, run_number)
    det_unfiltered_df.to_csv(det_filename)

    run_number = '0015'
    det_filtered_df, det_unfiltered_df = pre_kitti.get_video_detections(run_date, run_number=run_number, lidar_fov = lidar_fov, det_classes = det_classes, eliminate_redundant = True, confidence_threshold = 0.8)
    det_filename = '../data/{}_drive_{}_detections_filtered.csv'.format(run_date, run_number)
    det_filtered_df.to_csv(det_filename)
    det_filename = '../data/{}_drive_{}_detections_unfiltered.csv'.format(run_date, run_number)
    det_unfiltered_df.to_csv(det_filename)
