import math
import numpy as np
import util.calibration_kitti as cal_kitti
import util.calibration as cal_santa_clara


class Transform():
    def __init__(self, parent_data = None, data_type = 'kitti'):

        self.parent_data = parent_data
        if data_type == 'kitti':
            self.cal = cal_kitti
        elif data_type == 'santaclara':
            self.cal = cal_santa_clara
        else:
            print('wrong run_type given')

        self.y_horizon = self.cal.cal['Y_HORIZON']
        self.y_resolution = self.cal.cal['Y_RESOLUTION']
        self.x_center = self.cal.cal['X_CENTER']
        self.x_resolution = self.cal.cal['X_RESOLUTION']
        self.vfov = self.cal.cal['VFOV']
        self.hfov = self.cal.cal['HFOV']
        self.ht_camera = self.cal.cal['HT_CAMERA']
        self.width_car = self.cal.cal['WIDTH_CAR']
        self.length_car = self.cal.cal['LENGTH_CAR']
        self.wl_ratio = self.width_car / self.length_car
        self.edge_margin = 0

        self.seg_to_pixel_top = self.cal.SEG_TO_PIXEL_TOP
        self.seg_to_pixel_center = self.cal.SEG_TO_PIXEL_CENTER
        self.seg_to_pixel_left = self.cal.SEG_TO_PIXEL_LEFT
        self.seg_to_pixel_right = self.cal.SEG_TO_PIXEL_RIGHT
        self.seg_to_pixel_bottom = self.cal.SEG_TO_PIXEL_BOTTOM
        self.alpha, self.zprime = self._y2_to_alpha_and_zprime()


    '''
    Function name:  _y2_to_Zprime
    Inputs:         calibration variables
                    y_resolution, y_horizon, vfov, ht_camera
    Output:         a dictionary of Z' values for each value of y2 from 
                    y_horizon-1 to y_resolution
    '''
    def _y2_to_alpha_and_zprime(self):
        zprime = {}
        alpha = {}
        for y in range(self.y_resolution, self.y_horizon, -1):
            val = (y-self.y_horizon) * self.vfov / self.y_resolution
            alpha[y] = val
            zprime[y] = self.ht_camera / math.tan(val)
        return alpha, zprime

    '''
    Function name:  _point_to_distance
    Inputs:         y2 - the y2 pixel value of the bounding box 
                    xc - the x pixel value of the point of interest
                    yc - the y pixel value of the point of interest
                    
    Output:         distance to the point of interest 
    
    equation:       dist = zprime / cos(beta)
                    beta = sqrt((xc-x_center)^2 + (yc-ycenter)^2) / Y2    
    '''
    def _point_to_distance(self, y2, cent):
        if y2 > self.y_horizon and y2 <= self.y_resolution:
            dist_pixels = math.sqrt((cent[0] - self.x_center)**2 + (cent[1] - self.y_horizon)**2)
            beta = dist_pixels / (y2-self.y_horizon) * self.alpha[y2]
            dist = self.zprime[y2] / math.cos(beta)
            return dist
        else:
            return 100000 # if y2 is out of bounds return an abitrarily high number

    '''
    Function:   _x_to_seg_est
    Inputs:     x (in pixels)
    Output:     a floating point number whose integer value represents the number of the segment.
    '''
    def _x_to_seg_est(self,x):
        return (x - self.seg_to_pixel_left[0])/((self.seg_to_pixel_right[15]-self.seg_to_pixel_left[0])/self.parent_data.n_segs)

    '''
    Function name:  _find_seg_intersections
    Inputs:         bb [x1,y1,x2,y2]
    Output:         list of lists intersecting segment numbers
    '''
    def find_seg_intersections(self,bbs):
        segs_list = []
        for bb in bbs:
            toohigh = bb[3] < self.seg_to_pixel_top
            toolow = bb[1] > self.seg_to_pixel_bottom
            if not (toohigh or toolow):
                seg_left = int(self._x_to_seg_est(bb[0]))
                seg_right = self._x_to_seg_est(bb[2])
                segs = [x for i,x in enumerate(range(0,self.parent_data.n_segs)) if x >= seg_left and x < seg_right]
                segs_list.append(segs)
            else:
                segs_list.append([])
        return segs_list

    '''
    Function name:  _find_seg_centroids
    Inputs:         bb, seg
    Outputs:        a list of (xc, yc) tuples corresponding to the seg list
    Method:         the function finds the intersecting area of the bounding box and the 
                    bounding box and returns the centroid of the intersecting area
    '''
    def _find_seg_centroids(self,bb,segs):
        cents = []
        for seg in segs:
            xvals = [bb[0],bb[2],self.seg_to_pixel_left[int(seg*self.parent_data.seg_step)],self.seg_to_pixel_right[int(seg*self.parent_data.seg_step)]]
            xvals.sort()
            yvals = [bb[1],bb[3],self.seg_to_pixel_top,self.seg_to_pixel_bottom]
            yvals.sort()
            cents.append((int((xvals[1] + xvals[2])/2),int((yvals[1] + yvals[2])/2)))
        return cents

    '''
    Function name:  bb_to_dist_seg
    Inputs:         a list containing [x1, y1, x2, y2]
                    which is the bounding box coordinates in the pixel space
    Output:         a list containing a list of intersecting segments numbers 0-15 and a 
                    list of the corresponding distance estimates
    '''

    def bb_to_dist_seg_list(self,bb):
        dists_list = []
        segs_list = self.find_seg_intersections(bb)
        for i, segs in enumerate(segs_list):
            cents = self._find_seg_centroids(bb[i],segs_list[i])
            dists = []
            for cent in cents:
                dist = self._point_to_distance(bb[i][3],cent)
                dists.append(dist)
            dists_list.append(dists)
        return dists_list, segs_list

    '''
    Function name:  bb_dist_to_XZ
    Inputs:         a list of lists containing [x1, y1, x2, y2, dist]
                    which is the bounding box coordinates in the pixel space and
                    the dist in ft in the XZ space
    Output:         a list of lists of [X, Z] tuples for each list in the input
    '''

    def bb_dist_to_XZ(self,bb_dist_lists):
        XZ_lists = []
        for i, bb_dist_list in enumerate(bb_dist_lists):

            # x_c and y_c are the coordinates of the centroid of the bounding box in the pixel plane xy
            x_c = (bb_dist_list[0] + bb_dist_list[2]) / 2
            y_c = (bb_dist_list[1] + bb_dist_list[3]) / 2

            # hypotenus_xy is the distance in pixels from the center of optical flow
            # and the centroid of the bounding box in the pixel coordinate system
            hypotenus_xy = math.sqrt((x_c - self.x_center)**2 + (y_c - self.y_horizon)**2)

            # beta is the angle between the line extending from the camera to the center of optical flow
            # and the line extending from the camera to the center of the object in the world coordinate system
            beta = hypotenus_xy * self.hfov / self.x_resolution

            # Z is the Z coordinate of the object in the world coordinate system
            Z = bb_dist_list[4] * math.cos(beta)

            # hypotenus_XY is the distance in feet between the point (0, 0, Z) and (Xc, Yc, Z)
            # in the world coordinate system on the plane parallel to XY
            # that intersects the back of the detected object
            hypotenus_XY = bb_dist_list[4] * math.sin(beta)

            # X is the X coordinate of the object in the world coordinate system
            X = x_c / hypotenus_xy * hypotenus_XY
            XZ_lists.append([X,Z])

        return XZ_lists

    # this function calculates the ideal bounding box for the lidar detection
    def lidar_dist_seg_to_bb(self, dist, seg):

        y2 = int(self.cal.cal['FOCAL_LENGTH'] * self.cal.cal['HT_CAMERA'] / dist + self.cal.cal['Y_HORIZON'])

        x_width =  self.cal.cal['WIDTH_CAR'] / dist * self.cal.cal['FOCAL_LENGTH']

        beta = math.radians(self.cal.SEG_TO_ANGLE[seg*self.parent_data.seg_step])
        x_mid = self.cal.cal['X_CENTER'] + beta / self.cal.cal['HFOV'] * self.cal.cal['X_RESOLUTION']

        x1 = int(x_mid - x_width / 2)
        x2 = int(x_mid + x_width / 2)
        y1 = int(y2 - x_width)

        return [x1, y1, x2, y2]


if __name__ == '__main__':

    tr = Transform()

    alpha, zprime = tr._y2_to_alpha_and_zprime()

    print(tr.seg_to_pixel_left[0], tr.seg_to_pixel_top, tr.seg_to_pixel_right[15], tr.seg_to_pixel_bottom)

    bbs = [[100,200,400,400],[800,200,1100,400],[400,400,800,500]]

    dists_list, segs_list = tr.bb_to_dist_seg_list(bbs)

    print(dists_list)
