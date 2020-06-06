from src.detection.car_detector_tf_v2 import CarDetectorTFV2
import cv2
import glob
import os
from src.association.costs import Costs
import numpy as np
import pandas as pd
import src.util.calibration as cal


class Preprocess_SantaClara():
    def __init__(self):
        self.detector = CarDetectorTFV2()

    def avi_to_png(self, run_date, run_number):
        DATA_DIR = '../data'
        PAUSE = False

        video_feed = cv2.VideoCapture()
        video_feed.open('{}/{}/{}.avi'.format(DATA_DIR, run_date, run_number))

        frame_num = -1
        while True:
            if PAUSE is not True:
                success, frame = video_feed.read()
                if not success:
                    print('no frame')
                    break
                else:
                    frame_num += 1

                    cv2.imshow('draw_frame', frame)

                    filename = '{0:s}/{1:s}/{1:s}_drive_{2:s}_sync/image_02/data/{3:08.0f}.png'.format(DATA_DIR, run_date, run_number, frame_num)
                    cv2.imwrite(filename, frame)

            key = cv2.waitKey(30) & 0xFF
            if key == ord('q') or key == 27:
                exit(0)
            if key == ord('p') or key == ord('P'):
                PAUSE = not PAUSE

    def get_video_detections(self, run_date, run_number, lidar_fov = [0,0,1280,720], det_classes = [], eliminate_redundant = False, confidence_threshold = 0.7):

        DATA_DIR = '../data'
        pathname = '{0:s}/{1:s}/{1:s}_drive_{2:s}_sync/image_02/data/*.png'.format(DATA_DIR, run_date,run_number)

        cam2_files = sorted(glob.glob(pathname))

        lidar_left = lidar_fov[0]
        lidar_top = lidar_fov[1]
        lidar_right = lidar_fov[2]
        lidar_bottom = lidar_fov[3]
        det_classes = det_classes
        det_values_filtered = []
        det_values_unfiltered = []

        num_files = len(cam2_files)
        c = Costs(data_type='santaclara')

        for frame_num, frame_filename in enumerate(cam2_files):
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
                new_bbs = []; new_class_names = []; new_confidences = []
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



if __name__ == '__main__':

    ps = Preprocess_SantaClara()

    data_dir = '../data'
    run_date = '2018-09-18'

    run_number = '0002'
    filename = '{0:s}/{1:s}/{1:s}_drive_{2:s}_sync/image_02/data/{3:08.0f}.png'.format(data_dir, run_date, run_number,0)
    if not os.path.exists(filename):
        ps.avi_to_png(run_date, run_number)
        print('Done with:{}, {}'.format(run_date, run_number))

    run_number = '0003'
    filename = '{0:s}/{1:s}/{1:s}_drive_{2:s}_sync/image_02/data/{3:08.0f}.png'.format(data_dir, run_date, run_number,0)
    if not os.path.exists(filename):
        ps.avi_to_png(run_date, run_number)
        print('Done with:{}, {}'.format(run_date, run_number))

    run_date = '2018-09-20'
    run_number = '0002'
    filename = '{0:s}/{1:s}/{1:s}_drive_{2:s}_sync/image_02/data/{3:08.0f}.png'.format(data_dir, run_date, run_number,0)
    if not os.path.exists(filename):
        ps.avi_to_png(run_date, run_number)
        print('Done with:{}, {}'.format(run_date, run_number))

    run_date = '2018-12-17'
    run_number = '0002'
    filename = '{0:s}/{1:s}/{1:s}_drive_{2:s}_sync/image_02/data/{3:08.0f}.png'.format(data_dir, run_date, run_number,0)
    if not os.path.exists(filename):
        ps.avi_to_png(run_date, run_number)
        print('Done with:{}, {}'.format(run_date, run_number))

    run_number = '0003'
    filename = '{0:s}/{1:s}/{1:s}_drive_{2:s}_sync/image_02/data/{3:08.0f}.png'.format(data_dir, run_date, run_number,0)
    if not os.path.exists(filename):
        ps.avi_to_png(run_date, run_number)
        print('Done with:{}, {}'.format(run_date, run_number))

    lidar_fov = [cal.SEG_TO_PIXEL_LEFT[0], cal.SEG_TO_PIXEL_TOP, cal.SEG_TO_PIXEL_RIGHT[15], cal.SEG_TO_PIXEL_BOTTOM ]
    lidar_range = [30, 140]
    det_classes = ['car', 'truck', 'bus']

    run_date = '2018-09-18'
    run_number = '0002'
    det_filtered_df, det_unfiltered_df = ps.get_video_detections(run_date=run_date, run_number=run_number, lidar_fov = lidar_fov, det_classes = det_classes, eliminate_redundant = True, confidence_threshold = 0.8)
    det_filename = '../data/{}_drive_{}_detections_filtered.csv'.format(run_date, run_number)
    det_filtered_df.to_csv(det_filename)
    det_filename = '../data/{}_drive_{}_detections_unfiltered.csv'.format(run_date, run_number)
    det_unfiltered_df.to_csv(det_filename)

