from __future__ import print_function

# standard external modules
import cv2
import numpy as np
import pandas as pd
import math
import pykitti
import glob
import sys

# add parent directory to the path
sys.path.append('../src/')
# custom modules

from detection.detection import Detection
from lidar.lidar_detection import LIDAR_detection
from association.association import Association
from association.costs import Costs
from util.transform import Transform


class SantaClaraDataset():
    '''a class to contain the camera images for the Santa Clara dataset in the same form as the kitti dataset'''
    def __init__(self):
        self.cam2_files = []


class AssociationTest():
    '''A class to perform the association process within the viewer application

    This class is the main container that supports the association between lidar values and detected objects using
    a cost based approach and the Hungarian algorithm. There are three major methods that support this:

    load_dataset - this method loads the datafiles for the full dataset into pandas dataframes
    get_frame - this method extracts the data for the current_frame of video
    perform_association - this method performs the association process for each of the video frames as part of the
                get_frame call. This process is supported by the Costs class in costs.py and the Association class in
                association.py file.

    Note that throughout this class there are 5 different types of data:

    images, lidar, ground truth, detections and association data

    The images are pulled on a frame by frame basis by the "get_frame" method
    The lidar, ground_truth and detection data is pulled all at once from a file when the "load_dataset" method is called
    EAch of these full pandas dataframes is filtered t by the "get_frame" method to include only the current_frame
    data. This smaller dataset is returned to the GUI for display so that the GUI doesn't have to separately handle
    filtering of the data

    These are the names of the pandas dataframes for the lidar, ground truth and detection data:

    lidar_df  --- the full lidar dataframe
    lidar_frame --- the lidar data for just the current frame

    gt_df --- the full ground truth dataframe
    dt_frame --- the ground truth data for just the current frame

    det_df --- the full detection dataframe
    det_frame --- the detection data for just the current frame

    The associations data is created during execution of the perform_association method. The association data is
    containedt in a dictionary called association_frame It contains several outputs of the association process
    for the current video frame that gets returned to the GUI for display.

    Inside the association_frame dictionary are the following:

            self.association_frame = {
                'assignments'       :   self.assignments,
                'total_costs'       :   self.total_costs,
                'cost_components'   :   self.cost_components,
                'function_names'    :   self.function_names
            }

    assignments - a list of tuples containing the a pair (detection index, lidar_index) for each association
    total costs - the cost array that is input into the Hungarian Algorithm
    cost_components - a list of 4 cost arrays for each of the components of the total cost
    function_names - the names of the functions that were used for the calculation of the individual cost components

    '''

    def __init__(self, parent_data):
        self.parent_data = parent_data # this is the variable used to access the variables from the Frame class
        self.data_dir = self.parent_data.data_dir
        self.USE_NEW_GT = True
        self.USE_NEW_KITTI = False
        self.date = None
        self.drive = None

        # these are the relative paths to the data files in the data directory
        self.LIDAR_FILENAME_BASE = '{}/{}/{}_filtered_new.csv'
        self.TRACKLET_FILENAME_BASE = '{}/2011_09_26_drive_{}_sync_complete_tracklets.csv'
        self.DETECTION_FILENAME_BASE = '{}/{}_drive_{}_detections_unfiltered.csv'
        self.SANTA_CLARA_IMAGE_FILEPATH_BASE = '{}/{}/{}_drive_{}_sync/image_02/data/*.png'
        self.KITTI_POINTS_VISIBLE_CAM2 = '{}/{}/{}_clustered_points_vis2_1.csv'
        self.SANTA_CLARA_GROUND_TRUTH = '{}{}/viewer_logs/gt_records/gt_record_{}_{}.csv'

        # initialize the class variables
        self.dataset = None
        self.gt_df = None
        self.det_df = None
        self.lidar_df = None
        self.gt_frame = None
        self.det_frame = None
        self.lidar_frame = None
        self.gt_df_filtered = None
        self.det_df_filtered = None
        self.lidar_df_filtered = None
        self.association_frame = None
        self.image = None
        self.assignments = None
        self.total_costs = None
        self.cost_components = None
        self.function_names = None
        self.clustered_df = None
        self.clustered_df_frame = None
        self.clustered_segs_only = None

        return

    # helper methods to make filenames from the base paths defined above in the __init__ method
    def get_lidar_filename(self, date, drive):
        return self.LIDAR_FILENAME_BASE.format(self.data_dir, date, drive)

    def get_tracklet_filename(self,date,drive):
        return self.TRACKLET_FILENAME_BASE.format(self.data_dir, drive)

    def get_detection_filename(self, date, drive):
        return self.DETECTION_FILENAME_BASE.format(self.data_dir, date, drive)

    def get_santa_clara_image_filepath(self, date, drive):
        return self.SANTA_CLARA_IMAGE_FILEPATH_BASE.format(self.data_dir, date, date, drive)

    def get_kitti_points_visible_cam2(self, date, drive):
        return self.KITTI_POINTS_VISIBLE_CAM2.format(self.data_dir, date, drive)

    def get_santa_clara_ground_truth(self, date, drive):
        return self.SANTA_CLARA_GROUND_TRUTH.format(self.data_dir, date, date, drive)

    # filter lidar dataframe within min and max distance values
    def filter_lidar_df(self):
        self.lidar_df_filtered = self.lidar_df[self.lidar_df['distance'] >= self.parent_data.min_distance]
        self.lidar_df_filtered = self.lidar_df_filtered[self.lidar_df_filtered['distance'] <= self.parent_data.max_distance]
        return

    # filter the kitti ground truth values for min/max distance and object type
    def filter_gt_df(self):
        # remove all objects that are not vehicles
        if self.parent_data.run_types[self.parent_data.run_date] == 'kitti':
            self.gt_df_filtered = self.gt_df[(self.gt_df['label'].isin(self.parent_data.gt_object_classes))]
        elif self.parent_data.run_types[self.parent_data.run_date] == 'santaclara':
            self.gt_df_filtered = self.gt_df[(self.gt_df['detection_class'].isin(self.parent_data.det_object_classes))]
            # self.gt_df_filtered = self.gt_df_filtered.loc[self.gt_df_filtered['is_valid'] == True] # is_valid is filtered out during accuracy test
        else:
            print('bad run_type parameter: {}'.format(self.parent_data.run_types[self.parent_data.run_date]))

        if self.parent_data.run_types[self.parent_data.run_date] == 'kitti':
            # remove all objects that are closer than min_distance or farther than max_distance
            self.gt_df_filtered = self.gt_df_filtered[
                (self.gt_df_filtered['distance'] >= self.parent_data.min_distance)]
            self.gt_df_filtered = self.gt_df_filtered[
                (self.gt_df_filtered['distance'] <= self.parent_data.max_distance)]
        elif self.parent_data.run_types[self.parent_data.run_date] == 'santaclara':
            # remove all objects that are closer than min_distance or farther than max_distance
            self.gt_df_filtered = self.gt_df_filtered[(self.gt_df_filtered['lidar_distance'] >= self.parent_data.min_distance)]
            self.gt_df_filtered = self.gt_df_filtered[(self.gt_df_filtered['lidar_distance'] <= self.parent_data.max_distance)]
        else:
            print('bad run_type parameter: {}'.format(self.parent_data.run_types[self.parent_data.run_date]))
        return

    # filter the YOLO detections for object type and detection confidence
    def filter_det_df(self):
        # remove all objects that are not in the selected object detection classes
        self.det_df_filtered = self.det_df[(self.det_df['detection_class'].isin(self.parent_data.det_object_classes))]

        # if a min detection confidence threshold is defined, filter objects with lower confidence that the threshold
        if self.parent_data.enable_min_det_confidence:
            self.det_df_filtered = self.det_df_filtered[self.det_df_filtered['detection_confidence'] >= self.parent_data.min_det_confidence]

        return

    def load_dataset(self,date,drive):
        '''loads the complete datasets from files and puts these into pandas dataframes for subsequent processing

         This method is called whenever the drive date and run number values are changed on the GUI
         The entire dataset is read into several pandas dataframes for lidar, detection and ground truth (if kitti)
         The datasets are then filtered as specified by the GUI settings
         '''
        self.date = date
        self.drive = drive

        if self.parent_data.run_types[self.parent_data.run_date] == 'kitti':

            # create a class to access the kitti dataset
            self.dataset = pykitti.raw(base_path=self.data_dir, date=date, drive=drive)
            tracklet_filename = self.get_tracklet_filename(date, drive)
            self.gt_df = pd.read_csv(tracklet_filename)

            if self.USE_NEW_GT:
                for i in range(16):
                    self.gt_df['dist_seg{0:0.0f}'.format(i)] = self.gt_df['dist_seg{0:0.0f}'.format(i)] * self.parent_data.cal.cal['M_TO_FT']
            else:
                self.gt_df['dist'] = self.gt_df['dist'] * self.parent_data.cal.cal['M_TO_FT']
            self.load_kitti_m16_data(date,drive)

        elif self.parent_data.run_types[self.parent_data.run_date] == 'santaclara':

            # create a class to access the santa clara dataset
            self.dataset = SantaClaraDataset()
            filepath = self.get_santa_clara_image_filepath(date, drive)
            self.dataset.cam2_files = sorted(glob.glob(filepath))
            santa_clara_ground_truth_filename = self.get_santa_clara_ground_truth(date, drive)
            try:
                self.gt_df = pd.read_csv(santa_clara_ground_truth_filename)
                self.gt_df = self.gt_df.loc[self.gt_df['is_valid'] == True]
            except:
                self.gt_df = None
            self.clustered_df = None
        else:
            print('bad run_type parameter: {}'.format(self.parent_data.run_types[self.parent_data.run_date]))

        # read in the lidar data
        lidar_filename = self.get_lidar_filename(date, drive)
        self.lidar_df = pd.read_csv(lidar_filename, skiprows=2)
        self.lidar_df.drop(['unix_time', 'elapsed_time', 'fps', 'amplitude', 'flags'], axis=1, inplace=True)
        if self.parent_data.run_types[self.parent_data.run_date] == 'santaclara':
            self.lidar_df['distance'] = self.lidar_df['distance'] * self.parent_data.cal.cal['M_TO_FT']

        # read in the YOLO data
        detection_filename = self.get_detection_filename(date, drive)
        self.det_df = pd.read_csv(detection_filename)
        self.det_df.drop(['Unnamed: 0'], axis=1, inplace=True)

        # filter the data
        self.filter_lidar_df()
        self.filter_det_df()
        self.filter_gt_df()

        # reset the is_old flag so that it won't load again
        self.parent_data.isOld['dataset'] = False
        return

    # this function calculates the ideal bounding box for the lidar detection
    def lidar_dist_seg_to_bb(self, dist, seg):

        y2 = int(self.parent_data.cal.cal['FOCAL_LENGTH'] * self.parent_data.cal.cal['HT_CAMERA'] / dist + self.parent_data.cal.cal['Y_HORIZON'])

        x_width = self.parent_data.cal.cal['WIDTH_CAR'] / dist * self.parent_data.cal.cal['FOCAL_LENGTH']

        angle = 0
        for i in range(self.parent_data.seg_step):
            angle += self.parent_data.cal.SEG_TO_ANGLE[(seg * self.parent_data.seg_step) + i]

        angle /= self.parent_data.seg_step
        beta = math.radians(angle)
        x_mid = self.parent_data.cal.cal['X_CENTER'] + beta / self.parent_data.cal.cal['HFOV'] * self.parent_data.cal.cal['X_RESOLUTION']

        x1 = int(x_mid - x_width / 2)
        x2 = int(x_mid + x_width / 2)
        y1 = int(y2 - x_width)

        return [x1, y1, x2, y2]

    # helper functions used by pandas for filtering
    def ideal_bb(self, row): return self.lidar_dist_seg_to_bb(row.distance, row.segment)
    def x1_from_list(self, row): return row.bbox[0]
    def y1_from_list(self, row): return row.bbox[1]
    def x2_from_list(self, row): return row.bbox[2]
    def y2_from_list(self, row): return row.bbox[3]

    def get_frame(self, cam, frame, isFiltered, use_detector, weights, isOld, data_type):
        '''this function is called every time a new frame of video and data is requested by the application

        get_frame filters the data as required by the settings in the GUI and it also performs the association process
        on the data using the perform_associations method using the cost component weights and the max cost settings
        that are specified in the GUI
        '''

        if self.dataset is None or isOld['dataset']: # just in case - load the dataset
            self.load_dataset(self.parent_data.run_date, self.parent_data.run_number)
            isOld['dataset'] = False

        if isOld['ground_truth']:
            if self.parent_data.run_types[self.parent_data.run_date] == 'santaclara':
                santa_clara_ground_truth_filename = self.get_santa_clara_ground_truth(self.date, self.drive)
                try:
                    self.gt_df = pd.read_csv(santa_clara_ground_truth_filename)
                    self.gt_df = self.gt_df.loc[self.gt_df['is_valid'] == True]
                except:
                    self.gt_df = None
            isOld['ground_truth'] = False

        if isOld['image']: # get a new image if needed
            if cam == 0:
                image_filename = self.dataset.cam0_files[frame-self.parent_data.run_start_frame]
            elif cam == 1:
                image_filename = self.dataset.cam1_files[frame-self.parent_data.run_start_frame]
            elif cam == 2:
                image_filename = self.dataset.cam2_files[frame-self.parent_data.run_start_frame]
            elif cam == 3:
                image_filename = self.dataset.cam3_files[frame-self.parent_data.run_start_frame]
            else:
                image_filename = self.dataset.cam2_files[frame - self.parent_data.run_start_frame]

            image = cv2.imread(image_filename)

            self.image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            isOld['image'] = False

        if self.parent_data.run_types[self.parent_data.run_date] == 'kitti': # get the kitti data for the frame
            if isOld['gt_frame'] or isOld['filtered']:
                if isFiltered:
                    gt_df = self.gt_df_filtered
                else:
                    gt_df = self.gt_df

                self.gt_frame = gt_df.loc[gt_df['frame_number'] == frame].copy()
                self.gt_frame.insert(1, 'gt_index', 0)

                for i in range(len(self.gt_frame)):
                    self.gt_frame.iloc[i, 1] = i
                isOld['gt_frame'] = False

        elif self.parent_data.run_types[self.parent_data.run_date] == 'santaclara': # get the santa clara data for the frame
            if self.gt_df is None:
                self.gt_frame = pd.DataFrame([], columns=self.parent_data.gt_record_columns)
            else:
                self.gt_frame = self.gt_df.loc[self.gt_df['det_frame'] == frame].copy()

        else:
            print('wrong run_type given: {}'.format(self.parent_data.run_types[self.parent_data.run_date]))

        # TODO - add filter to eliminate duplicate detection classes of the same object

        # get the detection data for the frame
        if isOld['det_frame'] or isOld['filtered']:
            if isFiltered:
                det_df = self.det_df_filtered
                if isOld['filtered']:
                    self.filter_det_df()
            else:
                det_df = self.det_df
            self.det_frame = det_df.loc[det_df['frame'] == frame].copy()

            isOld['det_frame'] = False
            isOld['filtered'] = False

        # get the lidar data for the frame
        if isOld['lidar_frame'] or isOld['filtered']:
            if isFiltered:
                lidar_df = self.lidar_df_filtered
            else:
                lidar_df = self.lidar_df

            # adjust for the lag frames parameter - this is because the video is lagging the lidar data during capture
            self.lidar_frame = lidar_df.loc[lidar_df['frame'] == (frame - self.parent_data.lag_frames)].copy()

            # this line changes the lidar segments number based on the number of segments selected on the GUI
            self.lidar_frame['segment'] = self.lidar_frame['segment'] // self.parent_data.seg_step

            # put the ideal bounding box into columns in the lidar_frame dataframe
            if len(self.lidar_frame) > 0:
                self.lidar_frame['bbox'] = self.lidar_frame.apply(self.ideal_bb, axis=1)
                self.lidar_frame['x1'] = self.lidar_frame.apply(self.x1_from_list, axis=1)
                self.lidar_frame['y1'] = self.lidar_frame.apply(self.y1_from_list, axis=1)
                self.lidar_frame['x2'] = self.lidar_frame.apply(self.x2_from_list, axis=1)
                self.lidar_frame['y2'] = self.lidar_frame.apply(self.y2_from_list, axis=1)
                self.lidar_frame.drop(['bbox'], axis=1, inplace=True)
            else:
                self.lidar_frame['x1'] = 0
                self.lidar_frame['y1'] = 0
                self.lidar_frame['x2'] = 0
                self.lidar_frame['y2'] = 0

            # index the lidar values
            self.lidar_frame['lidar_index'] = 0
            self.lidar_frame = self.lidar_frame[['frame', 'lidar_index', 'segment', 'distance', 'cluster_label', 'x1', 'y1', 'x2', 'y2']]
            for i in range(len(self.lidar_frame)):
                self.lidar_frame.iloc[i,1] = i
            isOld['lidar_frame'] = False

        # call the perform_associations process
        if isOld['association']:
            self.assignments, self.total_costs, self.cost_components = self.perform_association(use_detector, weights, data_type=data_type)

            # put the results into a dictionary to be returned
            self.association_frame = {
                'assignments'       :   self.assignments,
                'total_costs'       :   self.total_costs,
                'cost_components'   :   self.cost_components,
                'function_names'    :   self.function_names
            }
            isOld['association'] = False

        # return the data for the current frame to the GUI
        return self.image, self.gt_frame, self.det_frame, self.lidar_frame, self.association_frame

    def perform_association(self, use_detector, weights, data_type):
        '''this is the method that does the association between the lidar values and the detected objects

        it uses several helper classes:
        Transform - a class that is used to transform between the image plane and the lidar distances
        Costs - a class that is used to contain the algorithms to compute the cost components
        Association - performs the assocation process
        Munkres - A third party method that does the Hungarian algorithm

        The perform_association process works like this:

        1) Create a list of detection objects using the Detection class
            a) if use_intersecting_only is checked - remove any detections that have no lidar values near them
        2) Create a list of lidar detectio objects using the LIDAR_detection class
            b) if use_intersecting_only is checked - remove any lidar detections that have no objects near them
        3) Create a list of cost component functions from the Costs class that will be used to calculate the costs

        4) Create an Association object and send these three lists into the Association class method - evaluate_cost

        5) Send this cost_matrix into the Association class method compute_munkres to get the assignments back of
                which lidar values got assigned to which objects

        6) If use_intersecting_only is checked - reassemble the lidar detections and the object detections back into
                their original ordering in a big cost matrix with -1's put into the locations where non-intersecting
                lidar values and detection objects reside.
        7) Return the assignments, the total_costs and the component_costs to the calling function

        '''

        tr = Transform(parent_data=self.parent_data, data_type=data_type)

        max_cost_assignments = []
        function_names = []

        # make a list of all of the lidar segments that have lidar values in them
        # (this is used for eliminating non-intersecting objects)
        lidar_segs = []
        for i in range(len(self.lidar_frame)):
            lidar_segs.append(self.lidar_frame.iloc[i, 2])

        # make a list of video_detections that will contain the individual Detection objects
        video_detections = []

        # make a list of the indexes of video detections so that the order can be reassembled when
        # the non-intersecting objects are put back into the cost matrices
        video_detections_idx = []

        # if the use_detector is False, we use the ground truth rather than the detection objects
        if use_detector:
            use_frame = self.det_frame
        else:
            use_frame = self.gt_frame

        # make a list to contain all of the segments that are intersected by object bounding boxes
        # (this will be used to eliminate non-intersecting lidar values)
        all_intersecting_segs = []

        # populate the video_detections list and the all_intersecting_segs list
        for i in range(len(use_frame)):
            video_det = Detection()
            video_det.bbox = [use_frame.iloc[i,4], use_frame.iloc[i,5], use_frame.iloc[i,6], use_frame.iloc[i,7]]
            if self.parent_data.use_intersecting_only:
                segs_list = tr.find_seg_intersections([video_det.bbox])[0]
                all_intersecting_segs = all_intersecting_segs + segs_list
                if len([x for x in segs_list if x in lidar_segs]) > 0:
                    video_detections.append(video_det)
                    video_detections_idx.append(i)
            else:
                video_detections.append(video_det)

        # make a list to contain the LIDAR_detection objects
        lidar_detections = []

        # make a list of the indexes of the lidar detections so that the order can be reassembled when
        # the non-intersecting lidar detections are put back into the cost matrices
        lidar_detections_idx = []

        # populate the lidar detections list
        for i in range(len(self.lidar_frame)):
            lidar_det = LIDAR_detection(0, self.lidar_frame.iloc[i, 2], self.lidar_frame.iloc[i, 3], 0,
                                        self.parent_data.cal, parent_data=self.parent_data)
            if self.parent_data.use_intersecting_only:
                if self.lidar_frame.iloc[i, 2] in all_intersecting_segs:
                    lidar_detections.append(lidar_det)
                    lidar_detections_idx.append(i)
            else:
                lidar_detections.append(lidar_det)

        # perform the associations task
        total_costs = []
        cost_components = []
        assignments = []
        if len(lidar_detections) > 0 and len(video_detections) > 0:
            costs = Costs(parent_data=self.parent_data, data_type=data_type)

            # total_cost(i,j) = w_0 * cost_function_0(i,j) + w_1 * cost_function_1(i,j) + .. + w_n * cost_function_n(i,j)

            cost_functions = {costs.dist_between_centroids: weights[0],
                              costs.dist_lidar_to_y2estimate: weights[1],
                              costs.inverse_intersection_over_union: weights[2],
                              costs.video_detection_intersects_segment: weights[3]}

            names = ['dist_between_centroids',
                     'dist_lidar_to_y2estimate',
                     'inverse_intersection_over_union',
                     'video_detection_intersects_segment']

            a = Association()

            # enter the video_detections and lidar_detections lists into the kwargs dictionary
            kwargs = {'video_detections': video_detections, 'lidar_detections': lidar_detections}

            # evaluate the costs array by passing the cost_functions dictionary
            # and the kwargs dictionary to the evaluate_costs method
            costs, cost_components = a.evaluate_cost(cost_functions, names, **kwargs)

            # make a deep copy to preserve the total costs (the munkres algorithm modifies the cost array)
            total_costs = costs.copy()

            # the munkres algorithm needs a matrix that is square or tall but not wide - so we transpose wide matrices
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

        # translate to original sizes if use_intersecting_only is checked
        if self.parent_data.use_intersecting_only and len(use_frame) > 0 and len(self.lidar_frame) > 0:
            new_total_costs = np.ones((len(use_frame), len(self.lidar_frame)), np.float32) * -1
            new_cost_components = []

            for i in range(4): # the number of cost components is 4
                # fill a matrix with all -1's
                new_cost_components.append(np.ones((len(use_frame), len(self.lidar_frame)), np.float32) * -1)
            if len(total_costs) > 0: # shape returns only one value if it is empty
                (m,n) = np.shape(total_costs)
            else:
                m = 0; n = 0
            # write over the -1's with the cost values for the intersecting objects and lidar values
            for i in range(m):
                for j in range(n):
                    new_total_costs[video_detections_idx[i], lidar_detections_idx[j]] = total_costs[i, j]
                    for k in range(len(cost_components)):
                        new_cost_components[k][video_detections_idx[i],lidar_detections_idx[j]] = cost_components[k][i,j]

            # fix the index values for the assignments to account for the non-intersecting values that were removed
            new_assignments = []
            for i in range(len(assignments)):
                assignment = (video_detections_idx[assignments[i][0]], lidar_detections_idx[assignments[i][1]])
                new_assignments.append(assignment)

            # overwrite the values to be returned
            total_costs = new_total_costs
            cost_components = new_cost_components
            assignments = new_assignments

        return assignments, total_costs, cost_components

    def evaluate_frame_association_accuracy(self, cam, frame, isFiltered, use_detector, weights, data_type):
        ncount = 0 # total possible associations
        ntp = 0    # true positives (correct associations)
        nfp = 0    # false positives (incorrect associations)
        nfn = 0    # false negatives (associations that were missed or rejected)
        frame_object_details = []

        # Get the frame data
        isOld = {
            'gt_frame'          : True,
            'det_frame'         : True,
            'lidar_frame'       : True,
            'association'       : True,
            'accuracy'          : False,
            'accuracy_settings' : False,
            'filtered'          : False,
            'image'             : False,
            'dataset'           : False,
            'clusters'          : False,
            'ground_truth'      : False
        }

        self.image, self.gt_frame, self.det_frame, self.lidar_frame, self.association_frame = self.get_frame(cam, frame, isFiltered, use_detector, weights, isOld, data_type)
        # go through the ground truth (gt) data and find determine if associations were made correctly
        # methodology:
        # step through each gt object
        #   1) if the the video_detection_index of the gt_frame row is in the first element of the association pairs
        #       we can conclude that an association was made for that object
        #       a) if the lidar distance associated with the detection object is within x% of the gt lidar distance
        #          then the association is a true positive (a correct association)
        #       b) if the lidar distance associated with the detection object is not within x% of gt lidar distance
        #          then the association is a false positive (an incorrect association)
        #   2) if the video_detection_index of the gt_frame row is not in the first elements of the association pairs
        #      then the association is a false negative ( an association that should have been made but wasn't)

        # step 1
        lidar_distance_tolerance = 0.05
        asmts = self.association_frame['assignments']
        for i in range(len(self.gt_frame)):
            if self.gt_df_filtered.iloc[i, :].is_valid:
                det_index = self.gt_frame.iloc[i, :].video_detection_index
                found = False
                for j in range(len(asmts)):
                    if det_index == asmts[j][0]:
                        found = True
                        gt_dist = self.gt_frame.iloc[i,:].lidar_distance
                        assoc_dist = self.lidar_frame.iloc[asmts[j][1], :].distance
                        if abs(gt_dist - assoc_dist) / gt_dist <= lidar_distance_tolerance:
                            ntp += 1   # step 1a
                            ncount += 1
                            frame_object_details.append([frame, i, det_index,'tp'])
                        else:
                            nfp += 1   # step 1b
                            ncount += 1
                            frame_object_details.append([frame, i, det_index, 'fp'])
                if not found:
                    nfn += 1    # step 2
                    ncount += 1
                    frame_object_details.append([frame, i, det_index, 'fn'])

        return ncount, ntp, nfp, nfn, frame_object_details

    def evaluate_run_association_accuracy(self, cam, isFiltered, use_detector, weights, data_type):
        count = 0    # total possible associations
        tpcount = 0  # true positives (correct associations)
        fpcount = 0  # false positives (incorrect associations)
        fncount = 0  # false negatives (associations that were missed or rejected)
        run_object_details = []
        valid_gt_frames = self.gt_df.loc[self.gt_df['is_valid']]['det_frame'].unique()
	#print("*elements in frame",len(valid_gt_frames)
        for frame_num in valid_gt_frames:
            ncount, ntp, nfp, nfn, frame_object_details = self.evaluate_frame_association_accuracy(cam, frame_num, isFiltered,use_detector,weights,data_type)
            count += ncount
            tpcount += ntp
            fpcount += nfp
            fncount += nfn
            run_object_details.extend(frame_object_details)

        accy = tpcount / count
        precision = tpcount / (tpcount + fpcount)
        recall = tpcount / (tpcount + fncount)
        run_stats = [accy, precision, recall]
        return count, tpcount, fpcount, fncount, run_stats, run_object_details

    # a helper function for the kitti dataset
    def load_kitti_m16_data(self, date, drive):
        filename = self.get_kitti_points_visible_cam2(date, drive)
        self.clustered_df = pd.read_csv(filename, skiprows=2)
        self.clustered_df.cluster_label = self.clustered_df.cluster_label.astype(int)

    # a helper function for the kitti dataset
    def get_kitti_frame(self, frame):
        if self.parent_data.isOld['clusters']:
            self.clustered_df_frame = self.clustered_df.loc[self.clustered_df['frame'] == frame]
            self.clustered_segs_only = self.clustered_df_frame.loc[(self.clustered_df_frame['segment'] >= 0) & (self.clustered_df_frame['cluster_label'] >= 0)]
            self.clustered_df_frame.sort_values(by=['cluster_label', 'segment'], inplace=True)

        self.parent_data.isOld['clusters'] = False
        return self.clustered_df_frame, self.clustered_segs_only

    def update_gt_df(self, record_gt_df):
        self.gt_df = record_gt_df.copy()
        return

