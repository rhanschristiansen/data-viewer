from __future__ import print_function
import os
import cv2
import numpy as np
import pandas as pd
from src.detection.car_detector_tf_v2 import CarDetectorTFV2
from src.detection.detection import Detection
from src.lidar.lidar_detection import LIDAR_detection
from src.association.association import Association
from src.association.costs import Costs
import src.util.calibration_kitti as cal
import math
import pykitti
import datetime
from tqdm import tqdm

WRITE_VIDEO_FILE = False
WRITE_DATA_FILE = False
USE_NEW_GT_FILE = True

USE_DETECTOR = True
CONFIDENCE_THRESHOLD = 0.8


global x_pixel, y_pixel, no_mouse_click_count
x_pixel = -1
y_pixel = -1
max_no_mouse_click_count = 100
no_mouse_click_count = max_no_mouse_click_count

def mouse_click(event, x, y, flags, param):
    global x_pixel, y_pixel, no_mouse_click_count
    no_mouse_click_count = max_no_mouse_click_count

    if event == cv2.EVENT_MOUSEMOVE:
        x_pixel = x
        y_pixel = y

PWD = os.path.dirname(__file__)
DATA_DATE = '2011_09_26'
RUN_NUMBER = '0015'

video_frame_lag = 0
max_cost = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # set higher to accept more assignments
dist_thresh = 0.2 # set higher to allow less accurate lidar readings to be labeled as correct
# w0: L2norm, w1: y2_est, w2: iou
weights = [0, 0.25, 0.75]
w0 = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
w1 = [0, 0.2, 0.4, 0.6, 0.8, 1.0]


# create a class to access the kitti dataset
kitti_dataset = pykitti.raw(base_path='../data/', date=DATA_DATE, drive=RUN_NUMBER)

DATA_DIR = '../data'

lidar_left = cal.SEG_TO_PIXEL_LEFT[0]
lidar_right = cal.SEG_TO_PIXEL_RIGHT[15]
lidar_top = cal.SEG_TO_PIXEL_TOP
lidar_bottom = cal.SEG_TO_PIXEL_BOTTOM

# read the lidar data
m16 = pd.read_csv('{}/{}/{}_filtered.csv'.format(DATA_DIR, DATA_DATE, RUN_NUMBER ),skiprows=2)

# read the ground truth from the tracklets data
if USE_NEW_GT_FILE:
    ## new ground truth format
    gt_df = pd.read_csv('{}/{}'.format(DATA_DIR, '2011_09_26_drive_'+RUN_NUMBER+'_sync_complete_tracklets.csv'))
    gt_df['dist'] = gt_df['dist'] * cal.cal['M_TO_FT']
    for i in range(16):
        colname = 'dist_seg{0:0.0f}'.format(i)
        gt_df[colname] = gt_df[colname] * cal.cal['M_TO_FT']
else:
    gt_df = pd.read_csv('{}/{}'.format(DATA_DIR, '2011_09_26_drive_' + RUN_NUMBER + '_sync_converted-tracklets.csv'))

# remove all objects that are not vehicles
gt_df = gt_df[(gt_df['label']=='Car') | (gt_df['label']=='Truck') | (gt_df['label']=='Van')]
# remove all objects outside the range of the lidar detector
gt_df = gt_df[(gt_df['dist']>=20) & (gt_df['dist'] <= 140)]
#remove all objects outside of the lidar fov
gt_df = gt_df[gt_df['x1'] <= lidar_right]
gt_df = gt_df[gt_df['x2'] >= lidar_left]
gt_df = gt_df[gt_df['y1'] <= lidar_bottom]
gt_df = gt_df[gt_df['y2'] >= lidar_top]

def draw_bboxes(bboxes, img):
    """
    Draw bounding boxes to frame
    :param bboxes: list of bboxes in [x1,y1,x2,y2] format
    :param img: np.array
    :return: image with bboxes drawn
    """
    img = img.copy()
    if bboxes is not None and len(bboxes) > 0:
        for i, bb in enumerate(bboxes):
            cv2.rectangle(img, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), 2)

    return img
if USE_DETECTOR:
    detector = CarDetectorTFV2()

PAUSE = False
DISP_LIDAR = False
DISP_DET = False
DISP_ASSOC = True
DISP_ZONES = True
DISP_TRUTH = True
DISP_RESULTS = True
SLOW = False


column_names_2 = ['run_num','use_detector', 'max_cost', 'w0', 'w1', 'w2', 'total_associations', 'accuracy', 'precision', 'recall', 'total_possible_associations', 'true_pos', 'false_pos', 'false_neg', 'false_pos_non_intersecting_bbox', 'false_pos_distance_error', 'false_neg_max_cost', 'false_neg_no_association', 'false_pos_association_no_gt']
#test_results = pd.read_csv('test_runs.csv')

manual_test = [0, False, 0.5, 0.9, 0, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
test_results = pd.DataFrame([manual_test], columns=column_names_2)

first_frame = True

run_num = 0

for run_num in tqdm(range(len(test_results))):

    total_possible_associations = 0
    true_pos = 0
    false_pos = 0
    false_neg = 0
    false_pos_non_intersecting_bbox = 0
    false_pos_distance_error = 0
    false_neg_max_cost = 0
    false_neg_no_association = 0
    false_pos_association_no_gt = 0

    USE_DETECTOR = test_results.loc[test_results['run_num'] == run_num].use_detector.bool()

    max_cost = np.float(test_results.loc[test_results['run_num'] == run_num].max_cost)
    w0 = np.float(test_results.loc[test_results['run_num'] == run_num].w0)
    w1 = np.float(test_results.loc[test_results['run_num'] == run_num].w1)
    w2 = np.float(test_results.loc[test_results['run_num'] == run_num].w2)

    weights = [w0, w1, w2]

    column_names = ['frame', 'video_det_index', 'lidar_det_index', 'gt_index', 'lidar_dist', 'gt_dist', 'cost',
                    'correct', 'max_cost', 'dist_thresh', 'w0', 'c0', 'w1', 'c1', 'w2', 'c2', ]
    associations_record = pd.DataFrame([], columns=column_names)

    for frame_num, frame_filename in enumerate(kitti_dataset.cam2_files):
        new_frame = True
        frame = cv2.imread(frame_filename)
        success = frame.any()
        frame_draw = frame.copy()
        if not success:
            print('no frame')
            break

        if first_frame:
            cv2.namedWindow('draw_frame')
            cv2.setMouseCallback('draw_frame', mouse_click)
            first_frame = False

        # get the ground truth values
        gt_current_frame = gt_df.loc[gt_df['frame_number'] == frame_num]


        # fill in the list of detections
        bboxes = []
        c = Costs(data_type='kitti')
        frame_list = [167,200,213,221,242,254,262,263,267,269,274,275,290]



        if frame_num in frame_list:
            a = 1

        if USE_DETECTOR:
            # get the video detections
            bbs, class_names, confidences = detector.detect(img=frame, return_class_scores=True)
            for i, bb in enumerate(bbs):
                # ensure bb is inside the window
                bbs[i][0] = max(bb[0],0)
                bbs[i][1] = max(bb[1],0)
                bbs[i][2] = min(bb[2],cal.cal['X_RESOLUTION'])
                bbs[i][3] = min(bb[3],cal.cal['Y_RESOLUTION'])
            n = len(class_names)
            eliminate_flag = np.zeros(n,np.int)
            for i, (bbox, class_name, confidence) in enumerate(zip(bbs, class_names, confidences)):
                # filter out bounding boxes that do not intersect with the lidar zone and are not vehicles
                if class_name not in ['car', 'truck', 'bus'] or confidence <= CONFIDENCE_THRESHOLD:
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

            #eliminate redundant detections - iou greater that 0.5 and different class_name
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
                bboxes.append(bbox)

        else:
            # get the bounding boxes from the ground truth data
            gt_dist = []
            for i, gt in enumerate(gt_current_frame.values):
                #append gt to bboxes (ensure it is inside the window)
                if USE_NEW_GT_FILE:
                    bboxes.append([max(gt[3],0), max(gt[4],0), min(gt[5],cal.cal['X_RESOLUTION']), min(gt[6],cal.cal['Y_RESOLUTION'])])
                    gt_dist.append(list(gt[66:]))
                else:
                    bboxes.append([max(gt[1],0), max(gt[2],0), min(gt[3],cal.cal['X_RESOLUTION']), min(gt[4],cal.cal['Y_RESOLUTION'])])
                    gt_dist.append(gt[6])
                total_possible_associations += 1

        video_detections = []
        if bboxes is not None and len(bboxes) > 0:
            for i, bb in enumerate(bboxes):
                det = Detection()
                det.bbox = np.array([bb[0], bb[1], bb[2], bb[3]])
                det.frame_id = frame_num
                video_detections.append(det)

        # get the lidar values
        lidar_vals = m16.loc[m16['frame'] == frame_num-video_frame_lag]

        lidar_detections = []
        for ii in range(len(lidar_vals)):
            if lidar_vals.iloc[ii,5] >= 30 and lidar_vals.iloc[ii,5] <= 140:
                lidar_detection = LIDAR_detection(frame_num,int(lidar_vals.iloc[ii,4]),lidar_vals.iloc[ii,5],lidar_vals.iloc[ii,6])

                lidar_detections.append(lidar_detection)

        # perform the associations task
        assignments = []
        if len(lidar_detections) > 0 and len(video_detections) > 0:
            associations = []
            costs = Costs(data_type='kitti')

            # total_cost(i,j) = w_0 * cost_function_0(i,j) + w_1 * cost_function_1(i,j) + .. + w_n * cost_function_n(i,j)
            cost_functions = {costs.dist_between_centroids: weights[0],
                              costs.dist_lidar_to_y2estimate: weights[1],
                              costs.inverse_intersection_over_union: weights[2]}

            a = Association()

            # enter the video_detections and lidar_detections lists into the kwargs dictionary
            kwargs = {'video_detections': video_detections, 'lidar_detections': lidar_detections}

            # evaluate the costs array by passing the cost_functions dictionary and the kwargs dictionary to the evaluate_costs method
            costs, cost_components = a.evaluate_cost(cost_functions, **kwargs)

            original_costs = costs.copy()

            c_shape = np.shape(costs)
            rows = c_shape[0]
            cols = c_shape[1]
            if rows <= cols:
                assignments = a.compute_munkres(costs)
            else:
                costs_T = np.transpose(costs)

                assignments_T = a.compute_munkres(costs_T)
                assignments = []

                for i, assignment in enumerate(assignments_T):
                    assignments.append((assignment[1],assignment[0]))

            if len(assignments) != min(len(video_detections), len(lidar_detections)):
                a = 1

        # evaluate the results
        if USE_DETECTOR:
            # first step is to match video detection bounding boxes with ground truth bounding boxes
            assignment = None
            gt_cost = None
            gt_assignments = []


            gt_cost0 = np.ones((len(gt_current_frame),len(assignments)),np.float32)*100
            gt_cost1 = np.ones((len(gt_current_frame),len(assignments)),np.float32)*10
            gt_cost2 = np.ones((len(gt_current_frame),len(assignments)),np.float32)*1

            for i, gt in enumerate(gt_current_frame.values):
                if USE_NEW_GT_FILE:
                    bb_gt = [gt[3], gt[4], gt[5], gt[6]]
                else:
                    bb_gt = [gt[1], gt[2], gt[3], gt[4]]
                for j, assignment in enumerate(assignments):
                    # cost0[i, j]
                    bb_v = video_detections[assignment[0]].bbox
                    iou = c._iou(bb_gt, bb_v)
                    if iou > 0:
                        gt_cost0[i, j] = 1 - iou
                    #cost1[i, j]
                    if original_costs[assignment[0], assignment[1]] < max_cost:
                        gt_cost1[i, j] = 0
                    # cost2[i, j]
                    if USE_NEW_GT_FILE:
                        seg = int(lidar_detections[assignment[1]].seg)
                        gt_dist = gt[66+seg]
                        dist_error = abs(lidar_detections[assignment[1]].dist - gt_dist)/gt_dist
                        gt_cost2[i, j] = dist_error
                    else:
                        dist_error = abs(lidar_detections[assignment[1]].dist - gt[6])/gt[6]
                        gt_cost2[i, j] = dist_error

            gt_cost = gt_cost0 + gt_cost1 + gt_cost2

            a_gt = Association()

            original_gt_costs = gt_cost.copy()

            c_shape = np.shape(gt_cost)
            rows = c_shape[0]
            cols = c_shape[1]
            if rows > 0 and cols > 0:
                if rows <= cols:
                    gt_assignments = a.compute_munkres(gt_cost)
                else:
                    gt_cost_T = np.transpose(gt_cost)

                    gt_assignments_T = a.compute_munkres(gt_cost_T)
                    gt_assignments = []

                    for i, gt_assignment in enumerate(gt_assignments_T):
                        gt_assignments.append((gt_assignment[1],gt_assignment[0]))
            else:
                gt_assignments = []

            gt_assigned = np.zeros(len(gt_current_frame),np.int) #assigned gt objects if unassigned gt objects label as false_neg
            assoc_assigned = np.zeros(len(assignments),np.int)
            for i, gt_assignment in enumerate(gt_assignments):
                total_possible_associations += 1
                gt_assigned[gt_assignment[0]] = 1
                assoc_assigned[gt_assignment[1]] = 1
                gt = gt_current_frame.iloc[gt_assignment[0]]
                assignment = assignments[gt_assignment[1]]
                video_detection = video_detections[assignment[0]]
                lidar_detection = lidar_detections[assignment[1]]
                cost_val = original_gt_costs[gt_assignment[0],gt_assignment[1]]
                print('Frame: {0:0.0f}, Cost: {1:0.4f}'.format(frame_num, cost_val))

                if USE_NEW_GT_FILE:
                    seg = int(lidar_detections[assignment[1]].seg)
                    gt_dist = gt[66+seg]
                else:
                    gt_dist = gt[6]

                if  gt_cost2[gt_assignment[0],gt_assignment[1]] <= dist_thresh:
                    new_row = [frame_num, assignment[0], assignment[1], i,
                               lidar_detections[assignment[1]].dist, gt_dist,
                               original_costs[assignment[0], assignment[1]], 'true_pos', max_cost,
                               dist_thresh, weights[0],
                               cost_components[0][assignment[0], assignment[1]], weights[1],
                               cost_components[1][assignment[0], assignment[1]], weights[2],
                               cost_components[2][assignment[0], assignment[1]]]
                    true_pos += 1
                elif cost_val < 10:
                    new_row = [frame_num, assignment[0], assignment[1], i,
                               lidar_detections[assignment[1]].dist, gt_dist,
                               original_costs[assignment[0], assignment[1]], 'false_pos_distance_error', max_cost,
                               dist_thresh, weights[0],
                               cost_components[0][assignment[0], assignment[1]], weights[1],
                               cost_components[1][assignment[0], assignment[1]], weights[2],
                               cost_components[2][assignment[0], assignment[1]]]
                    false_pos += 1
                    false_pos_distance_error += 1

                elif cost_val < 100:
                    new_row = [frame_num, assignment[0], assignment[1], i,
                               lidar_detections[assignment[1]].dist, gt_dist,
                               original_costs[assignment[0], assignment[1]], 'false_neg_max_cost', max_cost,
                               dist_thresh, weights[0],
                               cost_components[0][assignment[0], assignment[1]], weights[1],
                               cost_components[1][assignment[0], assignment[1]], weights[2],
                               cost_components[2][assignment[0], assignment[1]]]
                    false_neg += 1
                    false_neg_max_cost += 1
                else:
                    new_row = [frame_num, assignment[0], assignment[1], i,
                               lidar_detections[assignment[1]].dist, gt_dist,
                               original_costs[assignment[0], assignment[1]], 'false_pos_non_intersecting_bbox', max_cost,
                               dist_thresh, weights[0],
                               cost_components[0][assignment[0], assignment[1]], weights[1],
                               cost_components[1][assignment[0], assignment[1]], weights[2],
                               cost_components[2][assignment[0], assignment[1]]]
                    false_pos += 1
                    false_pos_non_intersecting_bbox += 1

                associations_record.loc[len(associations_record)] = new_row

            for i in range(len(gt_assigned)):
                if gt_assigned[i] == 0:
                    total_possible_associations += 1
                    new_row = [frame_num, -1, -1, i, -1, gt_dist, -1, 'false_neg_no_association', max_cost,
                               dist_thresh, weights[0], -1, weights[1], -1, weights[2], -1]
                    false_neg += 1
                    false_neg_no_association += 1
                    associations_record.loc[len(associations_record)] = new_row

            for i in range(len(assoc_assigned)):
                if assoc_assigned[i] == 0:
                    # TODO - fix error on this line drive=0009, use_detector=True, Frame = 178
                    new_row = [frame_num, assignment[0], assignment[1], i, lidar_detections[assignment[1]].dist,
                               -1, original_costs[assignment[0], assignment[1]], 'false_pos_association_no_gt', max_cost,
                               dist_thresh, weights[0], -1, weights[1], -1, weights[2], -1]
                    false_pos_association_no_gt += 1
                    associations_record.loc[len(associations_record)] = new_row

        else:
            cols = ['frame', 'video_det_index', 'lidar_det_index', 'gt_index', 'lidar_dist', 'gt_dist', 'cost', 'correct']
            for assignment in assignments:
                if original_costs[assignment[0],assignment[1]] < max_cost:
                    if USE_NEW_GT_FILE:
                        seg = int(lidar_detections[assignment[1]].seg)
                        dist = gt_dist[assignment[0]][seg]
                    else:
                        dist = gt_dist[assignment[0]]

                    dist_error = abs(lidar_detections[assignment[1]].dist - dist)/dist


                    if dist_error < dist_thresh:
                        new_row = [frame_num, assignment[0], assignment[1], i, lidar_detections[assignment[1]].dist,
                                   dist, original_costs[assignment[0], assignment[1]], 'true_pos', max_cost, dist_thresh,
                                   weights[0], cost_components[0][assignment[0], assignment[1]], weights[1],
                                   cost_components[1][assignment[0], assignment[1]], weights[2],
                                   cost_components[2][assignment[0], assignment[1]]]
                        true_pos += 1
                    else:
                        new_row = [frame_num, assignment[0], assignment[1], i, lidar_detections[assignment[1]].dist,
                                   dist, original_costs[assignment[0], assignment[1]], 'false_pos', max_cost, dist_thresh,
                                   weights[0], cost_components[0][assignment[0], assignment[1]], weights[1],
                                   cost_components[1][assignment[0], assignment[1]], weights[2],
                                   cost_components[2][assignment[0], assignment[1]]]
                        false_pos += 1
                else:
                    new_row = [frame_num, assignment[0], assignment[1], i, lidar_detections[assignment[1]].dist,
                               dist, original_costs[assignment[0], assignment[1]], 'false_neg_maxcost',
                               max_cost, dist_thresh,
                               weights[0], cost_components[0][assignment[0], assignment[1]], weights[1],
                               cost_components[1][assignment[0], assignment[1]], weights[2],
                               cost_components[2][assignment[0], assignment[1]]]
                    false_neg += 1
                associations_record.loc[len(associations_record)] = new_row

        #display the frame once if not PAUSE; continuously if PAUSE
        while new_frame or PAUSE:
            new_frame = False # only go through once unless PAUSE

            # draw vertical line in center of image
            cv2.line(frame_draw, (int(cal.cal['X_CENTER']), 0), (int(cal.cal['X_CENTER']), int(frame_draw.shape[0])),
                     (255, 0, 255), 1)
            cv2.line(frame_draw, (0, int(cal.cal['Y_HORIZON'])), (int(frame_draw.shape[1]), int(cal.cal['Y_HORIZON'])),
                     (255, 0, 255), 1)
            cv2.putText(frame_draw, 'frame: {0:0.0f}'.format(frame_num), (0, 25), 1, 2, (0, 0, 255), 2)

            if DISP_DET: # show video detections in green
                for video_detection in video_detections:
                    cv2.rectangle(frame_draw, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), (0, 255, 0), 2)

            if DISP_LIDAR: # show lidar ideal bounding boxes in yellow
                for lidar_detection in lidar_detections:
                    lidar_dist = lidar_detection.dist
                    bb = lidar_detection.bb
                    cv2.rectangle(img=frame_draw, pt1=(int(bb[0]), int(bb[1])), pt2=(int(bb[2]), int(bb[3])),
                                  color=(0, 255, 255), thickness=2)
                    cv2.putText(frame_draw, '{0:0.2f}'.format(lidar_dist), (int(bb[0]), int(bb[1])), 1, 1, (0, 0, 255), 2)

            if DISP_ASSOC: # show associations in blue and red connected by a yellow line
                if len(lidar_detections) > 0 and len(video_detections) > 0:
                    for assignment in assignments:
#                        if original_costs[assignment[0],assignment[1]] <= max_cost:
                        bb_v = video_detections[assignment[0]].bbox
                        dist_est = float(video_detections[assignment[0]].dist_est_y2[assignment[1]])
                        lidar_dist = lidar_detections[assignment[1]].dist
                        cv2.rectangle(img=frame_draw, pt1=(int(bb_v[0]),int(bb_v[1])), pt2=(int(bb_v[2]),int(bb_v[3])), color=(255,0,0), thickness=2)
#                        cv2.putText(frame_draw, '{0:0.0f}'.format(assignment[0]), (int(bb_v[0]),int(bb_v[1])), 1, 1, (255, 0, 0), 2)
#                        cv2.putText(frame_draw, '{0:0.2f}'.format(dist_est), (int(bb_v[0]-30), int(bb_v[3]+25)), 1, 1, (255, 0, 0), 2)
                        bb_l = lidar_detections[assignment[1]].bb
                        cv2.rectangle(img=frame_draw, pt1=(int(bb_l[0]),int(bb_l[1])), pt2=(int(bb_l[2]),int(bb_l[3])), color=(0,0,255), thickness=2)
#                        cv2.putText(frame_draw, '{0:0.0f}'.format(assignment[1]), (int(bb_l[0]),int(bb_l[1])), 1, 1, (0, 0, 255), 2)
                        cv2.putText(frame_draw, '{0:0.2f}'.format(lidar_dist), (int(bb_l[0]),int(bb_l[3])+25), 1, 1, (0, 0, 255), 2)
                        cv2.line(img=frame_draw, pt1=(int(bb_v[0]),int(bb_v[1])), pt2=(int(bb_l[0]),int(bb_l[1])), color=(0,255,255), thickness=2)

            if DISP_ZONES: # show the lidar zone boundaries in black
                y1 = int(cal.SEG_TO_PIXEL_TOP)
                y2 = int(cal.SEG_TO_PIXEL_BOTTOM)
                for i in range(16):
                    x = int(cal.SEG_TO_PIXEL_LEFT[i])
                    cv2.line(frame_draw, (x, y1), (x, y2), (0, 0, 0), thickness=1)
                    cv2.line(frame_draw, (x - 5, y1), (x + 5, y1), (0, 0, 0), thickness=1)
                    cv2.line(frame_draw, (x - 5, y2), (x + 5, y2), (0, 0, 0), thickness=1)

                x = int(cal.SEG_TO_PIXEL_RIGHT[i])
                cv2.line(frame_draw, (x, y1), (x, y2), (0, 0, 0), thickness=1)
                cv2.line(frame_draw, (x - 5, y1), (x + 5, y1), (0, 0, 0), thickness=1)
                cv2.line(frame_draw, (x - 5, y2), (x + 5, y2), (0, 0, 0), thickness=1)

            if DISP_TRUTH: # show the ground truth bboxes and dist
                for i in range(len(gt_current_frame)):
                    if USE_NEW_GT_FILE:
                        label, x1, y1, x2, y2, dist = gt_current_frame.iloc[i,2:8]
                    else:
                        x1, y1, x2, y2, label, dist = gt_current_frame.iloc[i, 1:]

                    if not (x2 < lidar_left or x1 > lidar_right or y1 > lidar_bottom or y2 < lidar_top):
                        cv2.rectangle(img=frame_draw, pt1=(int(x1), int(y1)), pt2=(int(x2), int(y2)),color=(0, 255, 0), thickness=2)
                        cv2.putText(frame_draw, '{0:0.1f}'.format(dist), (int(x1), int(y1)), 1, 1,(0, 255, 0), 2)

            # show the nouse coordinates
            if x_pixel >= 0 and no_mouse_click_count > 0:
                cv2.putText(frame_draw, '({0:0.0f}, {1:0.0f})'.format(x_pixel, y_pixel),
                            (cal.cal['X_RESOLUTION'] - 120, 15), 1, 1, (0, 0, 255), 2)
                no_mouse_click_count -= 1

            if DISP_RESULTS:
                pass

            cv2.putText(frame_draw, 'frame: {0:0.0f}'.format(frame_num), (0, 25), 1, 2, (0, 0, 255), 2)
            cv2.imshow('draw_frame', frame_draw)

            if SLOW:
                key = cv2.waitKey(1000) & 0xFF
            else:
                key = cv2.waitKey(30) & 0xFF

            if frame_num in frame_list:
                a = 1

            if key == ord('q') or key == 27:
                exit(0)
            if key == ord('p') or key == ord('P'):
                PAUSE = not PAUSE
            if key == ord('l') or key == ord('L'):
                DISP_LIDAR = not DISP_LIDAR
            if key == ord('d') or key == ord('D'):
                DISP_DET = not DISP_DET
            if key == ord('a') or key == ord('A'):
                DISP_ASSOC = not DISP_ASSOC
            if key == ord('s') or key == ord('S'):
                SLOW = not SLOW
            if key == ord('z') or key == ord('Z'):
                DISP_ZONES = not DISP_ZONES
            if key == ord('t') or key == ord('T'):
                DISP_TRUTH = not DISP_TRUTH

    accuracy = true_pos / total_possible_associations
    if true_pos + false_pos > 0:
        precision = true_pos / (true_pos + false_pos)
    else:
        precision = np.nan

    if true_pos + false_neg > 0:
        recall = true_pos / (true_pos + false_neg)
    else:
        recall = np.nan

    now = datetime.datetime.now()
    filename = 'results_{0:04d}.csv'.format(run_num)

    associations_record.to_csv(filename, index=False)

    print('run: {0:0.0f}, accy: {1:0.3f}, prec: {2:0.3f}, recall: {3:0.3f}, total_assoc: {4:0.0f}, \
           total_poss_assoc: {5:0.0f}, true_pos: {6:0.0f}, false_pos: {7:0.0f}, false_neg: {8:0.0f}, \
           false_pos_non_intersect_bb: {9:0.0f}, false_pos_distance_error: {10:0.0f}, \
           false_neg_max_cost: {11:0.0f}, false_neg_no_association: {12:0.0f}, false_pos_association_no_gt: {13:0.0f}, \
           use_detector:{14:}, max_cost: {15:0.3f}, w0: {16:0.3f}, w1: {17:0.3f}, w2: {18:0.3f}'.format(run_num,
           accuracy, precision, recall, len(associations_record), total_possible_associations, true_pos, false_pos,
           false_neg, false_pos_non_intersecting_bbox, false_pos_distance_error, false_neg_max_cost,
           false_neg_no_association, false_pos_association_no_gt, str(USE_DETECTOR), max_cost, weights[0], weights[1], weights[2]))

    test_results.iloc[run_num,6] = len(associations_record)
    test_results.iloc[run_num,7] = accuracy
    test_results.iloc[run_num,8] = precision
    test_results.iloc[run_num,9] = recall
    test_results.iloc[run_num,10] = total_possible_associations
    test_results.iloc[run_num,11] = true_pos
    test_results.iloc[run_num,12] = false_pos
    test_results.iloc[run_num,13] = false_neg
    test_results.iloc[run_num,14] = false_pos_non_intersecting_bbox
    test_results.iloc[run_num,15] = false_pos_distance_error
    test_results.iloc[run_num,16] = false_neg_max_cost
    test_results.iloc[run_num,17] = false_neg_no_association
    test_results.iloc[run_num,18] = false_pos_association_no_gt


filename = 'test_results_' + str(now) + '.csv'
test_results.to_csv(filename)
print('Run Complete!')
