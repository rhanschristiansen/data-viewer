import math
import util.calibration_kitti as cal_kitti
import util.calibration as cal_santaclara
import numpy as np
from util.transform import Transform

# the class Costs contains methods that individually calculate the cost_arrays for use by the Munkres algorithm
# Each of the methods receives the lidar_detections and video_detections lists as values in the **kwargs dictionary
# Each of the methods returns a cost array with values designed to be between 0 and 1
# the cost array should have the number of rows equal to the number of video detection objects in the video_detections list
# and the number of columns equal to the number of lidar detection objects in the lidar_detections list.

class Costs():
    def __init__(self, parent_data = None, data_type = 'kitti'):
        self.parent_data = parent_data
        if data_type == 'kitti':
            self.cal = cal_kitti
        elif data_type == 'santaclara':
            self.cal = cal_santaclara
        else:
            print('wrong datatype given')
        self.tr = Transform(parent_data=self.parent_data, data_type=data_type)

    # this method calculates the euclidian distance between the centroid of the video_detection bounding box and the
    # centroid of the lidar detection ideal bounding box. The distance in pixels is divided by the diagonal of the
    # video image in pixels to give a value between 0 and 1
    def dist_between_centroids(self, **kwargs):
#        max_dist = math.sqrt(cal.cal['X_RESOLUTION']**2 + cal.cal['Y_RESOLUTION']**2) # max dist between centroids used
        LIDAR_X_RES = self.cal.SEG_TO_PIXEL_RIGHT[8] - self.cal.SEG_TO_PIXEL_LEFT[0]
        LIDAR_Y_RES = self.cal.SEG_TO_PIXEL_BOTTOM - self.cal.SEG_TO_PIXEL_TOP

        max_dist = math.sqrt(LIDAR_X_RES**2 + LIDAR_Y_RES**2) # max dist between centroids used to normalize between 0 and 1
        video_detections = kwargs['video_detections']
        lidar_detections = kwargs['lidar_detections']
        dist_array = np.ones((len(video_detections), len(lidar_detections)), np.float64) * 1

        for i, video_detection in enumerate(video_detections):
            cx_v = (video_detection.bbox[0] + video_detection.bbox[2]) / 2
            cy_v = (video_detection.bbox[1] + video_detection.bbox[3]) / 2
            for j, lidar_detection in enumerate(lidar_detections):
                cx_l = (lidar_detection.bb[0] + lidar_detection.bb[2]) / 2
                cy_l = (lidar_detection.bb[1] + lidar_detection.bb[3]) / 2
                dist_array[i,j] = math.sqrt( (cx_v-cx_l)**2 + (cy_v-cy_l)**2 ) / max_dist

        return dist_array

    # this method calculates the difference in feet between the lidar distance and the
    # estimated distance derived from the y2 value of the video bounding box. this value is
    # divided by the max_lidar distance of 140 feet to give a value between 0 and 1
    # if there is no overlap between the video_detection bounding box and the lidar segment
    # the cost is penalized with a value of 1e6

    # TODO - Frame 2105 Object index 0 Lidar index 10 - y2 cost = 2.1296
    def dist_lidar_to_y2estimate(self, **kwargs):
        max_dist = 140/2 # maximum detection distance for lidar - used to normalize the output between 0 and 1
        m_to_ft = self.cal.cal['M_TO_FT']
        video_detections = kwargs['video_detections']
        lidar_detections = kwargs['lidar_detections']
        dist_array = np.ones((len(video_detections), len(lidar_detections)), np.float64) * 1
        bbs = []
        for i in range(len(video_detections)):
            bbs.append(list(video_detections[i].bbox))

        dists_list, segs_list = self.tr.bb_to_dist_seg_list(bbs)

        for i in range(len(dists_list)):
            dist_est_array = np.zeros((len(lidar_detections), 1), np.float64)
            for j in range(len(lidar_detections)):
                for segs, dists in zip(segs_list,dists_list):
                    for k, seg in enumerate(segs):
                        if seg == lidar_detections[j].seg: # only use values from bounding boxes that overlap the lidar segment
                            dist_est = dists[k]
                            # add dist_est to fvec in video_detection for later use
                            dist_est_array[j,0] = dist_est
                            dist_array[i,j] = abs(dist_est - lidar_detections[j].dist) / max_dist
            video_detections[i].dist_est_y2 = dist_est_array

        return dist_array

    # this is a helper function to calculate the intersection / union ratio of the bounding box rectangles.
    # the value 0 means there is no intersection
    # the value 1 means they bounding boxes are in exactly the same place 100% overlap
    def _iou(self, boxA, boxB):

        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        if xA < xB and yA < yB: # calculate area only for overlapping bounding boxes
            # compute the area of intersection rectangle
            interArea = (xB - xA) * (yB - yA)
        else:
            interArea = 0

        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        union = float(boxAArea + boxBArea - interArea)

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        if union > 0:
            iou = interArea / float(boxAArea + boxBArea - interArea)
        else:
            iou = 0

        return iou

    # this method calculates the overlap of the video_detection bounding box and the
    # lidar detection ideal bounding box
    # The returned values in the array are 1 minus the intersection / union ratio
    # a 100% overlap would have a cost of 0 and 1% overlap would have a cost of 0.99
    # no overlap is penalized with a value of 1e6
    def inverse_intersection_over_union(self, **kwargs):

        video_detections = kwargs['video_detections']
        lidar_detections = kwargs['lidar_detections']

        cost_array = np.ones((len(video_detections), len(lidar_detections)), np.float64) * 1

        for i in range(len(video_detections)):
            for j in range(len(lidar_detections)):
                cost = 1 - self._iou(video_detections[i].bbox, lidar_detections[j].bb) # one minus the iou ratio
                if cost < 1: # only use values that are less that 1 (have some overlap between bounding boxes)
                    cost_array[i,j] = cost

        return cost_array

    def video_detection_intersects_segment(self, **kwargs):

        video_detections = kwargs['video_detections']
        lidar_detections = kwargs['lidar_detections']

        cost_array = np.ones((len(video_detections), len(lidar_detections)), np.float64) * 1

        for i, video_detection in enumerate(video_detections):
            bbs = video_detection.bbox
            segs_list = self.tr.find_seg_intersections([bbs])

            for j, lidar_detection in enumerate(lidar_detections):
                if lidar_detection.seg in segs_list[0]:
                    cost_array[i,j] = 0

        return cost_array
