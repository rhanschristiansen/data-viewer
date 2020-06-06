from association.association_test import AssociationTest
import util.calibration as cal_santaclara
import util.calibration_kitti as cal_kitti
import os
import cv2
import wx


def initialize_gui_controls(self):
    self.m_checkBox_show_coord.SetValue(self.show_coord)
    if self.use_detector:
        self.m_choice_run_use_detector.SetSelection(0)
    else:
        self.m_choice_run_use_detector.SetSelection(1)
    self.m_checkBox_use_filtered_data.SetValue(self.use_filtered_data)
    self.m_textCtrl_classes.SetValue(', '.join(self.det_object_classes))
    self.m_textCtrl_min_det_confidence.SetValue('{0:0.03f}'.format(self.min_det_confidence))
    self.m_checkBox_enable_min_det_confidence.SetValue(self.enable_min_det_confidence)
    self.m_checkBox_show_index_numbers.SetValue(self.show_index_numbers)
    self.m_checkBox_show_kitti_points.SetValue(self.show_kitti_points)
    self.m_checkBox_show_detection_zones.SetValue(self.show_detection_zones)
    self.m_checkBox_show_video_detections.SetValue(self.show_video_detections)
    self.m_checkBox_show_lidar_detections.SetValue(self.show_lidar_detections)
    self.m_checkBox_show_ground_truth.SetValue(self.show_ground_truth)
    self.m_checkBox_show_associations.SetValue(self.show_associations)
    self.m_checkBox_show_lidar_values.SetValue(self.show_lidar_values)
    self.m_colourPicker_detection_zones.SetColour(self.detection_zone_color)
    self.m_colourPicker_video_detections.SetColour(self.video_detections_color)
    self.m_colourPicker_lidar_detections.SetColour(self.lidar_detections_color)
    self.m_colourPicker_ground_truth.SetColour(self.ground_truth_color)
    self.m_colourPicker_associations.SetColour(self.associations_color)
    self.m_colourPicker_lidar_values_color.SetColour(self.lidar_value_color)
    self.m_textCtrl_logging_filepath.SetValue(self.logging_filepath)
    self.m_checkBox_enable_logging.SetValue(self.enable_logging)
    self.m_checkBox_enable_save_png.SetValue(self.enable_save_png)
    self.m_textCtrl_goto_frame_number.SetValue(str(self.goto_frame_number))
    self.m_choice_lag_frames.SetSelection(self.lag_frames_idx)
    self.m_textCtrl_max_cost.SetValue(str(self.max_cost))
    self.m_checkBox_enable_max_cost.SetValue(self.enable_max_cost)
    self.m_textCtrl_l2_norm_weight.SetValue(str(self.l2_norm_weight))
    self.m_textCtrl_y2_est_weight.SetValue(str(self.y2_est_weight))
    self.m_textCtrl_iou_weight.SetValue(str(self.iou_weight))
    self.m_textCtrl_seg_intersect_weight.SetValue(str(self.seg_intersect_weight))
    self.m_checkBox_use_intersecting_only.SetValue(self.use_intersecting_only)
    self.m_staticText_frame_total_possible.SetLabelText(str(self.frame_total_possible))
    self.m_staticText_frame_true_positives.SetLabelText(str(self.frame_true_positives))
    self.m_staticText_frame_false_positives.SetLabelText(str(self.frame_false_positives))
    self.m_staticText_frame_false_negatives.SetLabelText(str(self.frame_false_negatives))
    self.m_staticText_run_total.SetLabelText(str(self.run_total))
    self.m_staticText_run_true_positives.SetLabelText(str(self.run_true_positives))
    self.m_staticText_run_false_positives.SetLabelText(str(self.run_false_positives))
    self.m_staticText_run_false_negatives.SetLabelText(str(self.run_false_negatives))
    self.m_staticText_run_accuracy.SetLabelText('Accy: {0:0.4f}'.format(self.run_accuracy))
    self.m_staticText_run_precision.SetLabelText('Prec: {0:0.4f}%'.format(self.run_precision))
    self.m_staticText_run_recall.SetLabelText('Rcall: {0:0.4f}%'.format(self.run_recall))
    self.m_textCtrl_min_distance.SetValue(str(self.min_distance))
    self.m_textCtrl_max_distance.SetValue(str(self.max_distance))
    self.m_checkBox_enable_record_gt.SetValue(self.enable_record_gt)
    self.m_textCtrl_record_gt_filepath.SetValue(self.record_gt_filepath)


def update_run_date(self):
    self.m_choice_run_date.AppendItems(self.run_dates)
    self.m_choice_run_date.Select(self.run_date_idx)
    update_run_number(self)
    self.run_date = self.run_dates[self.run_date_idx]


def update_run_number(self):
    run_number_list = self.run_numbers[self.run_dates[self.run_date_idx]]
    self.m_choice_run_number.SetItems(run_number_list)
    if self.run_number_idx >= len(run_number_list):
        self.run_number_idx = 0
    self.m_choice_run_number.Select(self.run_number_idx)
    self.run_number = self.run_numbers[self.run_date][self.run_number_idx]


def load_dataset(self):
    self.current_frame = 0
    # display the initial image
    self.cal = update_cal(self)

    # instantiate the AssociationTest class
    self.at = AssociationTest(parent_data=self)
    self.at.min_distance = self.min_distance
    self.at.max_distance = self.max_distance
    self.at.load_dataset(date=self.run_date, drive=self.run_number)
    min_frame = int(self.at.dataset.cam2_files[0][-12:-4])
    max_frame = int(self.at.dataset.cam2_files[-1][-12:-4])

    self.run_start_frame = min_frame
    self.current_frame = min_frame
    self.run_stop_frame = max_frame

    # get the current frame data including the image, ground_truth, detections, lidar and associations
    self.image, self.gt_frame, self.det_frame, self.lidar_frame, self.association_frame = self.at.get_frame(2, self.current_frame, self.use_filtered_data, self.use_detector, self.weights, self.isOld, self.run_types[self.run_date])

    # get clustered data only for kitti
    if self.run_types[self.run_date] == 'kitti':
        self.clustered_df_frame, self.clustered_segs_only = self.at.get_kitti_frame(self.current_frame)
    elif self.run_types[self.run_date] == 'santaclara':
        self.clustered_df_frame = None
        self.clustered_segs_only = None
    else:
        print('wrong run_type given: {}'.format(self.run_types[self.run_date]))

    self.image_rows, self.image_cols, x = self.image.shape
    self.bmp = wx.Bitmap.FromBuffer(self.image_cols, self.image_rows, self.image)


def update_cal(self):
    if self.run_types[self.run_date] == 'kitti':
        cal = cal_kitti
    elif self.run_types[self.run_date] == 'santaclara':
        cal = cal_santaclara
    else:
        cal = cal_santaclara
        print('wrong run_type given: {}'.format(self.run_types[self.run_date]))
    return cal


def enable_disable_features(self, run_type):
    if run_type == 'santaclara':
        self.m_button_kitti_viewer.Disable()
        self.m_checkBox_show_ground_truth.SetValue(False)
        self.show_ground_truth = False
#        self.m_checkBox_show_ground_truth.Disable()
        self.m_checkBox_show_3d_ground_truth.SetValue(False)
        self.show_3d_ground_truth = False
        self.m_checkBox_show_3d_ground_truth.Disable()
        self.m_checkBox_show_kitti_points.SetValue(False)
        self.show_kitti_points = False
        self.m_checkBox_show_kitti_points.Disable()
        self.m_checkBox_show_all_kitti_points.Disable()
        self.m_choice_run_use_detector.Disable()

    elif run_type == 'kitti':
        self.m_button_kitti_viewer.Enable()
        self.m_checkBox_show_ground_truth.Enable()
        self.m_checkBox_show_3d_ground_truth.Enable()
        self.m_checkBox_show_kitti_points.Enable()
        if self.show_kitti_points:
            self.m_checkBox_show_all_kitti_points.Enable()
        else:
            self.m_checkBox_show_all_kitti_points.Disable()
    else:
        print('wrong run_type parameter: {}'.format(run_type))


def update_weights(self):
    '''A helper method to get updates to the weights from the GUI'''
    try:
        if self.l2_norm_weight != float(self.m_textCtrl_l2_norm_weight.GetValue()):
            self.l2_norm_weight = float(self.m_textCtrl_l2_norm_weight.GetValue())
            self.isOld['accuracy_settings'] = True
        if self.y2_est_weight != float(self.m_textCtrl_y2_est_weight.GetValue()):
            self.y2_est_weight = float(self.m_textCtrl_y2_est_weight.GetValue())
            self.isOld['accuracy_settings'] = True
        if self.iou_weight != float(self.m_textCtrl_iou_weight.GetValue()):
            self.iou_weight = float(self.m_textCtrl_iou_weight.GetValue())
            self.isOld['accuracy_settings'] = True
        if self.seg_intersect_weight != float(self.m_textCtrl_seg_intersect_weight.GetValue()):
            self.seg_intersect_weight = float(self.m_textCtrl_seg_intersect_weight.GetValue())
            self.isOld['accuracy_settings'] = True
        if self.max_cost != float(self.m_textCtrl_max_cost.GetValue()):
            self.max_cost = float(self.m_textCtrl_max_cost.GetValue())
            self.isOld['accuracy_settings'] = True
    except:
        print('Error reading input cell')
    self.weights = [self.l2_norm_weight, self.y2_est_weight, self.iou_weight, self.seg_intersect_weight]

def save_image_to_png(self):
    if not os.path.exists(self.logging_filepath):
        os.mkdir(self.logging_filepath)
    if self.last_frame_png_saved != self.current_frame:
        filepath_png = os.path.join(self.logging_filepath, 'image_{0:0.0f}.png'.format(self.current_frame))

        status = cv2.imwrite(filepath_png, cv2.cvtColor(self.display_image, cv2.COLOR_BGR2RGB))
        if status:
            self.last_frame_png_saved = self.current_frame


def log_to_file(self):
    filepath_assoc = os.path.join(self.logging_filepath, 'assoc.csv')
    filepath_lidar = os.path.join(self.logging_filepath, 'lidar.csv')
    filepath_detection = os.path.join(self.logging_filepath, 'detection.csv')
    files = [filepath_assoc, filepath_lidar, filepath_detection]
    if not os.path.exists(self.logging_filepath):
        os.mkdir(self.logging_filepath)
    if self.last_frame_logged == -1: # -1 is a sentenel value when the logging is first enabled.
        # write the settings to a log file
        filepath_settings = os.path.join(self.logging_filepath, 'settings.csv')
        settings_header = 'num_segments, max_cost, l2_norm_weight, y2_est_weight, iou_weight, seg_intersect_weight,is_filtered,object_classes, min_det_confidence, use_intersecting_only, min_distance, max_distance, lag_frames\n'
        settings = '{0:0.0f},{1:0.4f},{2:0.4f},{3:0.4f},{4:0.4f},{5:0.4f},{6:s},{7:s},{8:0.4f},{9:s},{10:0.0f},{11:0.0f},{12:0.0f}\n'.format(self.n_segs,
                    self.max_cost, self.l2_norm_weight, self.y2_est_weight, self.iou_weight, self.seg_intersect_weight,
                    str(self.use_filtered_data), str(self.det_object_classes), self.min_det_confidence,
                    str(self.use_intersecting_only), self.min_distance, self.max_distance, self.lag_frames)
        fp = open(filepath_settings, 'w')
        fp.write(settings_header)
        fp.write(settings)
        fp.close()

        # create files for associations, lidar and detections and write thier headers
        assoc_header = 'frame, assoc_index, assigned, segment, distance, object_index, lidar_index, total_cost, l2norm_cost, y2est_cost, iou_cost, seg_intersect_cost\n'
        lidar_header = 'frame, lidar_index, segment, distance, x1, y1, x2, y2\n'
        detection_header = 'frame, det_index, det_class, det_confidence, x1, y1, x2, y2\n'
        headers = [assoc_header, lidar_header, detection_header]
        for file, header in zip(files, headers):
            fp = open(file, 'w')
            fp.write(header)
            fp.close()

    # log the current frame to file
    if self.last_frame_logged != self.current_frame: # check that the frame hasn't already been written
        asmts = self.association_frame['assignments']
        det_idx = []
        for asmt in asmts:
            det_idx.append(asmt[0]) # put the detection object index into a list for logging the assoc_index number

        for i in range(len(self.det_frame)):
            # write each detection to file
            det_log_str = '{0:0.0f},{1:0.0f},{2:s},{3:0.4f},{4:0.0f},{5:0.0f},{6:0.0f},{7:0.0f}\n'.format(self.current_frame,
                            self.det_frame.iloc[i,1], self.det_frame.iloc[i,2], self.det_frame.iloc[i,3], self.det_frame.iloc[i,4],
                            self.det_frame.iloc[i,5], self.det_frame.iloc[i,6], self.det_frame.iloc[i,7])
            fp = open(filepath_detection, 'a')
            fp.write(det_log_str)
            fp.close()
            if i == 0:
                lidar_file_logged = False   # used so the lidar file will only be logged once
            for j in range(len(self.lidar_frame)):
                # write each lidar value to file (only first time through the values of j)
                if not lidar_file_logged:
                    lidar_log_str = '{0:0.0f},{1:0.0f},{2:0.0f},{3:0.5f},{4:0.0f},{5:0.0f},{6:0.0f},{7:0.0f}\n'.format(self.lidar_frame.iloc[j,0],
                                self.lidar_frame.iloc[j,1], self.lidar_frame.iloc[j,2], self.lidar_frame.iloc[j,3], self.lidar_frame.iloc[j,5],
                                self.lidar_frame.iloc[j,6], self.lidar_frame.iloc[j,7], self.lidar_frame.iloc[j,8])
                    fp = open(filepath_lidar, 'a')
                    fp.write(lidar_log_str)
                    fp.close()
                if not i in det_idx:
                    assoc_index = -1
                else:
                    assoc_index = det_idx.index(i)
                if assoc_index >= 0:
                    if (asmts[assoc_index][0] == i) and (asmts[assoc_index][1] == j):
                        assigned = 1
                    else:
                        assigned = 0
                else:
                    assigned = 0

                assoc_log_str = '{0:0.0f},{1:0.0f},{2:0.0f},{3:0.0f},{4:0.5f},{5:0.0f},{6:0.0f},{7:0.4f},{8:0.4f},{9:0.4f},{10:0.4f},{11:0.4f}\n'.format(self.current_frame,
                                assoc_index, assigned, self.lidar_frame.iloc[j,2], self.lidar_frame.iloc[j,3], i, j,
                                self.association_frame['total_costs'][i,j],
                                self.association_frame['cost_components'][0][i,j],
                                self.association_frame['cost_components'][1][i,j],
                                self.association_frame['cost_components'][2][i,j],
                                self.association_frame['cost_components'][3][i,j])
                fp = open(filepath_assoc, 'a')
                fp.write(assoc_log_str)
                fp.close()
            lidar_file_logged = True
    self.last_frame_logged = self.current_frame


def set_is_old(isOld, criteria, list=None):
    '''helper function  to set is_old dictionary values'''
    if criteria == 'all_true':
        isOld['gt_frame'] = True
        isOld['det_frame'] = True
        isOld['lidar_frame'] = True
        isOld['association'] = True
        isOld['accuracy'] = True
        isOld['accuracy_settings'] = True,
        isOld['filtered'] = True
        isOld['image'] = True
        isOld['dataset'] = True
        isOld['clusters'] = True

    elif criteria == 'list_true':
        for item in list:
            isOld[item] = True

    elif criteria == 'except_list_true':
        keys = ['gt_frame', 'det_frame', 'lidar_frame', 'association', 'accuracy', 'accuracy_settings', 'filtered', 'image', 'dataset', 'clusters']
        for key in keys:
            if not key in list:
                isOld[key] = True
    return