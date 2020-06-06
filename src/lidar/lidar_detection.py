import math

# the LIDAR_detection class is instantiated for each lidar reading that is reported by the lidar detector
# for every lidar reading in a frame there is an instance of this class
class LIDAR_detection():
    # the __init__ method is called when each instance of the LIDAR_detection object is created
    def __init__(self, frame, seg, dist, ampl, cal, parent_data = None):
        self.cal = cal                  # this is a reference to the calibration information in either calibration.py
                                        # or calibration_kitti.py it is stored in a dictionary
        self.parent_data = parent_data  # this is a reference to the class that instantiated the LIDAR_detection
                                        # object - it is needed so that we can access the number of segments displayed in the viewer
        self.frame = frame
        self.dist = dist
        self.seg = seg

        self.bb = self.lidar_dist_seg_to_bb(dist, seg) # this call creates the ideal bounding box from the distance and segment information
        self.ampl = ampl

    # this function calculates the ideal bounding box for the lidar detection
    def lidar_dist_seg_to_bb(self, dist, seg):

        # the calibration information is accessed by the keys ('FOCAL_LENGTH', 'HT_CAMERA' and 'Y_HORIZON')
        y2 = int(self.cal.cal['FOCAL_LENGTH'] * self.cal.cal['HT_CAMERA'] / dist + self.cal.cal['Y_HORIZON'])

        x_width =  self.cal.cal['WIDTH_CAR'] / dist * self.cal.cal['FOCAL_LENGTH']

        angle = 0
        for i in range(self.parent_data.seg_step):
            angle += self.parent_data.cal.SEG_TO_ANGLE[(seg * self.parent_data.seg_step) + i]

        angle /= self.parent_data.seg_step

        beta = math.radians(angle)
        x_mid = self.parent_data.cal.cal['X_CENTER'] + beta / self.parent_data.cal.cal['HFOV'] * self.parent_data.cal.cal['X_RESOLUTION']


#        beta = math.radians(self.cal.SEG_TO_ANGLE[seg*self.parent_data.seg_step])
#        x_mid = self.cal.cal['X_CENTER'] + (beta / self.cal.cal['HFOV'] * self.cal.cal['X_RESOLUTION']) * self.parent_data.seg_step

        x1 = int(x_mid - x_width / 2)
        x2 = int(x_mid + x_width / 2)
        y1 = int(y2 - x_width)

        return [x1, y1, x2, y2]

