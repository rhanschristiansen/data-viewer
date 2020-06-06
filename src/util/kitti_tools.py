import numpy as np
import pandas as pd
import pykitti
import cv2
from PIL import Image
from matplotlib import pyplot as plt
from src.source import parseTrackletXML as xmlParser
from mpl_toolkits.mplot3d import Axes3D
from src.source.utilities import print_progress
from moviepy.editor import ImageSequenceClip
from tqdm import tqdm
import os
import src.util.calibration_kitti as cal
import math


class Kitti_Tools():
    def __init__(self):
        # Change this to the directory where you store KITTI data
        self.basedir = '../../data'

        self.colors = {
            'Car': 'b',
            'Tram': 'r',
            'Cyclist': 'g',
            'Van': 'c',
            'Truck': 'm',
            'Pedestrian': 'y',
            'Sitter': 'k'
        }
        self.axes_limits = [
            [-20, 80],  # X axis range
            [-20, 20],  # Y axis range
            [-3, 10]  # Z axis range
        ]
        self.axes_str = ['X', 'Y', 'Z']

        self.VFOV_MIN = math.degrees(-cal.cal['VFOV']/2)
        self.VFOV_MAX = math.degrees(cal.cal['VFOV']/2)
        self.HFOV_MIN = math.degrees(-cal.cal['HFOV']/2)
        self.HFOV_MAX = math.degrees(cal.cal['HFOV']/2)
        self.VFOV = (self.VFOV_MIN, self.VFOV_MAX)
        self.HFOV = (self.HFOV_MIN, self.HFOV_MAX)
        self.VFOV = (-24.9, 2)
        self.HFOV = (-180, 180)
        self.SEGMENTS_LEFT = {}
        self.SEGMENTS_RIGHT = {}
        self.SEG_LENGTH = 400
        for i in range(16):
            angle_left = cal.SEG_TO_ANGLE_LEFT[i]
            angle_right = cal.SEG_TO_ANGLE_RIGHT[i]
            x_l = math.cos(math.radians(angle_left))*self.SEG_LENGTH
            y_l = math.sin(math.radians(-angle_left))*self.SEG_LENGTH
            x_r = math.cos(math.radians(angle_right))*self.SEG_LENGTH
            y_r = math.sin(math.radians(-angle_right))*self.SEG_LENGTH
            self.SEGMENTS_LEFT[i] = [0,0,x_l,y_l]
            self.SEGMENTS_RIGHT[i] = [0, 0, x_r, y_r]
        self.v2c_filepath = '../../data/2011_09_26/calib_velo_to_cam.txt'
        self.c2c_filepath = '../../data/2011_09_26/calib_cam_to_cam.txt'

    def load_dataset(self, date, drive, calibrated=False, frame_range=None):
        """
        Loads the dataset with `date` and `drive`.

        Parameters
        ----------
        date        : Dataset creation date.
        drive       : Dataset drive.
        calibrated  : Flag indicating if we need to parse calibration data. Defaults to `False`.
        frame_range : Range of frames. Defaults to `None`.

        Returns
        -------
        Loaded dataset of type `raw`.
        """
        dataset = pykitti.raw(self.basedir, date, drive)

        # Load the data
        if calibrated:
            dataset.load_calib()  # Calibration data are accessible as named tuples

        np.set_printoptions(precision=4, suppress=True)
        print('\nDrive: ' + str(dataset.drive))
        print('\nFrame range: ' + str(dataset.frames))

        if calibrated:
            print('\nIMU-to-Velodyne transformation:\n' + str(dataset.calib.T_velo_imu))
            print('\nGray stereo pair baseline [m]: ' + str(dataset.calib.b_gray))
            print('\nRGB stereo pair baseline [m]: ' + str(dataset.calib.b_rgb))

        return dataset

    def load_tracklets_for_frames(self, n_frames, xml_path):
        """
        Loads dataset labels also referred to as tracklets, saving them individually for each frame.

        Parameters
        ----------
        n_frames    : Number of frames in the dataset.
        xml_path    : Path to the tracklets XML.

        Returns
        -------
        Tuple of dictionaries with integer keys corresponding to absolute frame numbers and arrays as values. First array
        contains coordinates of bounding box vertices for each object in the frame, and the second array contains objects
        types as strings.
        """
        tracklets = xmlParser.parseXML(xml_path)

        frame_tracklets = {}
        frame_tracklets_types = {}
        for i in range(n_frames):
            frame_tracklets[i] = []
            frame_tracklets_types[i] = []

        # loop over tracklets
        for i, tracklet in enumerate(tracklets):
            # this part is inspired by kitti object development kit matlab code: computeBox3D
            h, w, l = tracklet.size
            # in velodyne coordinates around zero point and without orientation yet
            trackletBox = np.array([
                [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2],
                [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
                [0.0, 0.0, 0.0, 0.0, h, h, h, h]
            ])
            # loop over all data in tracklet
            for translation, rotation, state, occlusion, truncation, amtOcclusion, amtBorders, absoluteFrameNumber in tracklet:
                # determine if object is in the image; otherwise continue
                if truncation not in (xmlParser.TRUNC_IN_IMAGE, xmlParser.TRUNC_TRUNCATED):
                    continue
                # re-create 3D bounding box in velodyne coordinate system
                yaw = rotation[2]  # other rotations are supposedly 0
                assert np.abs(rotation[:2]).sum() == 0, 'object rotations other than yaw given!'
                rotMat = np.array([
                    [np.cos(yaw), -np.sin(yaw), 0.0],
                    [np.sin(yaw), np.cos(yaw), 0.0],
                    [0.0, 0.0, 1.0]
                ])
                cornerPosInVelo = np.dot(rotMat, trackletBox) + np.tile(translation, (8, 1)).T
                frame_tracklets[absoluteFrameNumber] = frame_tracklets[absoluteFrameNumber] + [cornerPosInVelo]
                frame_tracklets_types[absoluteFrameNumber] = frame_tracklets_types[absoluteFrameNumber] + [
                    tracklet.objectType]

        return (frame_tracklets, frame_tracklets_types)

    def draw_box(self, pyplot_axis, vertices, axes=[0, 1, 2], color='black'):
        """
        Draws a bounding 3D box in a pyplot axis.

        Parameters
        ----------
        pyplot_axis : Pyplot axis to draw in.
        vertices    : Array 8 box vertices containing x, y, z coordinates.
        axes        : Axes to use. Defaults to `[0, 1, 2]`, e.g. x, y and z axes.
        color       : Drawing color. Defaults to `black`.
        """
        vertices = vertices[axes, :]
        connections = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # Lower plane parallel to Z=0 plane
            [4, 5], [5, 6], [6, 7], [7, 4],  # Upper plane parallel to Z=0 plane
            [0, 4], [1, 5], [2, 6], [3, 7]  # Connections between upper and lower planes
        ]
        for connection in connections:
            pyplot_axis.plot(*vertices[:, connection], c=color, lw=0.5)


    def display_frame_statistics(self, dataset, tracklet_rects, tracklet_types, frame, points=0.2):
        """
        Displays statistics for a single frame. Draws camera data, 3D plot of the lidar point cloud data and point cloud
        projections to various planes.

        Parameters
        ----------
        dataset         : `raw` dataset.
        tracklet_rects  : Dictionary with tracklet bounding boxes coordinates.
        tracklet_types  : Dictionary with tracklet types.
        frame           : Absolute number of the frame.
        points          : Fraction of lidar points to use. Defaults to `0.2`, e.g. 20%.
        """
        dataset_gray = list(dataset.gray)
        dataset_rgb = list(dataset.rgb)
        dataset_velo = list(dataset.velo)

        print('Frame timestamp: ' + str(dataset.timestamps[frame]))
        # Draw camera data
        f, ax = plt.subplots(2, 2, figsize=(15, 5))
        ax[0, 0].imshow(dataset_gray[frame][0], cmap='gray')
        ax[0, 0].set_title('Left Gray Image (cam0)')
        ax[0, 1].imshow(dataset_gray[frame][1], cmap='gray')
        ax[0, 1].set_title('Right Gray Image (cam1)')
        ax[1, 0].imshow(dataset_rgb[frame][0])
        ax[1, 0].set_title('Left RGB Image (cam2)')
        ax[1, 1].imshow(dataset_rgb[frame][1])
        ax[1, 1].set_title('Right RGB Image (cam3)')
        plt.show()

        points_step = int(1. / points)
        point_size = 0.01 * (1. / points)
        velo_range = range(0, dataset_velo[frame].shape[0], points_step)
        velo_frame = dataset_velo[frame][velo_range, :]

        def draw_point_cloud(ax, title, axes=[0, 1, 2], xlim3d=None, ylim3d=None, zlim3d=None):
            """
            Convenient method for drawing various point cloud projections as a part of frame statistics.
            """
            ax.scatter(*np.transpose(velo_frame[:, axes]), s=point_size, c=velo_frame[:, 3], cmap='gray')
            ax.set_title(title)
            ax.set_xlabel('{} axis'.format(self.axes_str[axes[0]]))
            ax.set_ylabel('{} axis'.format(self.axes_str[axes[1]]))
            if len(axes) > 2:
                ax.set_xlim3d(*self.axes_limits[axes[0]])
                ax.set_ylim3d(*self.axes_limits[axes[1]])
                ax.set_zlim3d(*self.axes_limits[axes[2]])
                ax.set_zlabel('{} axis'.format(self.axes_str[axes[2]]))
            else:
                ax.set_xlim(*self.axes_limits[axes[0]])
                ax.set_ylim(*self.axes_limits[axes[1]])
            # User specified limits
            if xlim3d != None:
                ax.set_xlim3d(xlim3d)
            if ylim3d != None:
                ax.set_ylim3d(ylim3d)
            if zlim3d != None:
                ax.set_zlim3d(zlim3d)

            for t_rects, t_type in zip(tracklet_rects[frame], tracklet_types[frame]):
                self.draw_box(ax, t_rects, axes=axes, color=self.colors[t_type])

        # Draw point cloud data as 3D plot
        f2 = plt.figure(figsize=(15, 8))
        ax2 = f2.add_subplot(111, projection='3d')
        draw_point_cloud(ax2, 'Velodyne scan', xlim3d=(-10, 30))
        plt.show()

        # Draw point cloud data as plane projections
        f, ax3 = plt.subplots(3, 1, figsize=(15, 25))
        draw_point_cloud(
            ax3[0],
            'Velodyne scan, XZ projection (Y = 0), the car is moving in direction left to right',
            axes=[0, 2]  # X and Z axes
        )
        draw_point_cloud(
            ax3[1],
            'Velodyne scan, XY projection (Z = 0), the car is moving in direction left to right',
            axes=[0, 1]  # X and Y axes
        )
        draw_point_cloud(
            ax3[2],
            'Velodyne scan, YZ projection (X = 0), the car is moving towards the graph plane',
            axes=[1, 2]  # Y and Z axes
        )
        plt.show()


    def draw_3d_plot(self, frame, dataset, tracklet_rects, tracklet_types, points=0.2):
        """
        Saves a single frame for an animation: a 3D plot of the lidar data without ticks and all frame trackelts.
        Parameters
        ----------
        frame           : Absolute number of the frame.
        dataset         : `raw` dataset.
        tracklet_rects  : Dictionary with tracklet bounding boxes coordinates.
        tracklet_types  : Dictionary with tracklet types.
        points          : Fraction of lidar points to use. Defaults to `0.2`, e.g. 20%.

        Returns
        -------
        Saved frame filename.
        """
        dataset_velo = list(dataset.velo)

        f = plt.figure(figsize=(12, 8))
        axis = f.add_subplot(111, projection='3d', xticks=[], yticks=[], zticks=[])

        points_step = int(1. / points)
        point_size = 0.01 * (1. / points)
        velo_range = range(0, dataset_velo[frame].shape[0], points_step)
        velo_frame = dataset_velo[frame][velo_range, :]
        axis.scatter(*np.transpose(velo_frame[:, [0, 1, 2]]), s=point_size, c=velo_frame[:, 3], cmap='gray')
        axis.set_xlim3d(*self.axes_limits[0])
        axis.set_ylim3d(*self.axes_limits[1])
        axis.set_zlim3d(*self.axes_limits[2])
        for t_rects, t_type in zip(tracklet_rects[frame], tracklet_types[frame]):
            self.draw_box(axis, t_rects, axes=[0, 1, 2], color=self.colors[t_type])
        filename = 'video/frame_{0:0>4}.png'.format(frame)
        plt.savefig(filename)
        plt.close(f)
        return filename


    def prepare_animation_frames(self, n_frames):
        frames = []
        print('Preparing animation frames...')
        for i in range(n_frames):
            print_progress(i, n_frames - 1)
            filename = self.draw_3d_plot(i, dataset, tracklet_rects, tracklet_types)
            frames += [filename]
        print('...Animation frames ready.')
        return frames


    # These functions taken from here:
    # https://stackoverflow.com/questions/45333780/kitti-velodyne-point-to-pixel-coordinate
    def prepare_velo_points(self, pts3d_raw):
        '''Replaces the reflectance value by 1, and tranposes the array, so
           points can be directly multiplied by the camera projection matrix'''

        pts3d = pts3d_raw
        # Reflectance > 0
        pts3d = pts3d[pts3d[:, 3] > 0 ,:]
        pts3d[:,3] = 1
        return pts3d.transpose()

    def project_velo_points_in_img(self, pts3d, T_cam_velo, Rrect, Prect, min_z=0):
        '''Project 3D points into 2D image. Expects pts3d as a 4xN
           numpy array. Returns the 2D projection of the points that
           are in front of the camera only an the corresponding 3D points.'''
        #import pdb; pdb.set_trace()
        # 3D points in camera reference frame.
        pts3d_cam = Rrect.dot(T_cam_velo.dot(pts3d))

        # Before projecting, keep only points with z>0
        # (points that are in fronto of the camera).
        idx = (pts3d_cam[2,:]>=min_z)
        pts2d_cam = Prect.dot(pts3d_cam[:, idx])
        return pts3d[:, idx], pts2d_cam/pts2d_cam[2,:]

    # taken from https://github.com/charlesq34/frustum-pointnets/blob/2ffdd345e1fce4775ecb508d207e0ad465bcca80/kitti/kitti_util.py#L275
    def project_to_image(self, pts_3d, P):
        ''' Project 3d points to image plane.
        Usage: pts_2d = projectToImage(pts_3d, P)
          input: pts_3d: nx3 matrix
                 P:      3x4 projection matrix
          output: pts_2d: nx2 matrix
          P(3x4) dot pts_3d_extended(4xn) = projected_pts_2d(3xn)
          => normalize projected_pts_2d(2xn)
          <=> pts_3d_extended(nx4) dot P'(4x3) = projected_pts_2d(nx3)
              => normalize projected_pts_2d(nx2)
        '''
        n = pts_3d.shape[0]
        pts_3d_extend = np.hstack((pts_3d, np.ones((n,1))))
        print(('pts_3d_extend shape: ', pts_3d_extend.shape))
        pts_2d = np.dot(pts_3d_extend, np.transpose(P)) # nx3
        pts_2d[:,0] /= pts_2d[:,2]
        pts_2d[:,1] /= pts_2d[:,2]
        return pts_2d[:,0:2]

    def depth_color(self, val, min_d=0, max_d=120):
        """
        print Color(HSV's H value) corresponding to distance(m)
        close distance = red , far distance = blue
        """
        np.clip(val, 0, max_d, out=val)  # max distance is 120m but usually not usual
        return (((val - min_d) / (max_d - min_d)) * 120).astype(np.uint8)

    def in_h_range_points(self, points, m, n, fov):
        """ extract horizontal in-range points """
        return np.logical_and(np.arctan2(n, m) > (-fov[1] * np.pi / 180), \
                              np.arctan2(n, m) < (-fov[0] * np.pi / 180))

    def in_v_range_points(self, points, m, n, fov):
        """ extract vertical in-range points """
        return np.logical_and(np.arctan2(n, m) < (fov[1] * np.pi / 180), \
                              np.arctan2(n, m) > (fov[0] * np.pi / 180))

    def fov_setting(self, points, x, y, z, dist, h_fov, v_fov, filter_out_of_range=True):
        """ filter points based on h,v FOV  """

        if filter_out_of_range:

            if h_fov[1] == 180 and h_fov[0] == -180 and v_fov[1] == 2.0 and v_fov[0] == -24.9:
                return points

            if h_fov[1] == 180 and h_fov[0] == -180:
                return points[self.in_v_range_points(points, dist, z, v_fov)]
            elif v_fov[1] == 2.0 and v_fov[0] == -24.9:
                return points[self.in_h_range_points(points, x, y, h_fov)]
            else:
                h_points = self.in_h_range_points(points, x, y, h_fov)
                v_points = self.in_v_range_points(points, dist, z, v_fov)
                return points[np.logical_and(h_points, v_points)]
        else:
            return points

    def in_range_points(self, points, size):
        """ extract in-range points """
        return np.logical_and(points > 0, points < size)

    def velo_points_filter(self, points, v_fov, h_fov, filter_out_of_range=True):
        """ extract points corresponding to FOV setting """

        # Projecting to 2D
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        dist = np.sqrt(x ** 2 + y ** 2 + z ** 2)

        if h_fov[0] < -90:
            h_fov = (-90,) + h_fov[1:]
        if h_fov[1] > 90:
            h_fov = h_fov[:1] + (90,)

        x_lim = self.fov_setting(x, x, y, z, dist, h_fov, v_fov, filter_out_of_range=filter_out_of_range)[:, None]
        y_lim = self.fov_setting(y, x, y, z, dist, h_fov, v_fov, filter_out_of_range=filter_out_of_range)[:, None]
        z_lim = self.fov_setting(z, x, y, z, dist, h_fov, v_fov, filter_out_of_range=filter_out_of_range)[:, None]

        # Stack arrays in sequence horizontally
        xyz_ = np.hstack((x_lim, y_lim, z_lim))
        xyz_ = xyz_.T

        # stack (1,n) arrays filled with the number 1
        one_mat = np.full((1, xyz_.shape[1]), 1)
        xyz_ = np.concatenate((xyz_, one_mat), axis=0)

        # need dist info for points color
        if filter_out_of_range:
            dist_lim = self.fov_setting(dist, x, y, z, dist, h_fov, v_fov)
            color = self.depth_color(dist_lim, 0, 70)
        else:
            color = self.depth_color(dist,min(dist), max(dist))

        return xyz_, color

    def calib_velo2cam(self, filepath):
        """
        get Rotation(R : 3x3), Translation(T : 3x1) matrix info
        using R,T matrix, we can convert velodyne coordinates to camera coordinates
        """
        with open(filepath, "r") as f:
            file = f.readlines()

            for line in file:
                (key, val) = line.split(':', 1)
                if key == 'R':
                    R = np.fromstring(val, sep=' ')
                    R = R.reshape(3, 3)
                if key == 'T':
                    T = np.fromstring(val, sep=' ')
                    T = T.reshape(3, 1)
        return R, T

    def calib_cam2cam(self, filepath, mode):
        """
        If your image is 'rectified image' :
            get only Projection(P : 3x4) matrix is enough
        but if your image is 'distorted image'(not rectified image) :
            you need undistortion step using distortion coefficients(5 : D)

        in this code, I'll get P matrix since I'm using rectified image
        """
        with open(filepath, "r") as f:
            file = f.readlines()

            for line in file:
                (key, val) = line.split(':', 1)
                if key == ('P_rect_' + mode):
                    P_ = np.fromstring(val, sep=' ')
                    P_ = P_.reshape(3, 4)
                    # erase 4th column ([0,0,0])
                    P_ = P_[:3, :3]
        return P_

    def velo3d_2_camera2d_points(self, points, v_fov, h_fov, vc_path, cc_path, mode='02', filter_out_of_range=True):
        """ print velodyne 3D points corresponding to camera 2D image """
        # R_vc = Rotation matrix ( velodyne -> camera )
        # T_vc = Translation matrix ( velodyne -> camera )
        R_vc, T_vc = self.calib_velo2cam(vc_path)

        # P_ = Projection matrix ( camera coordinates 3d points -> image plane 2d points )
        P_ = self.calib_cam2cam(cc_path, mode)

        """
        xyz_v - 3D velodyne points corresponding to h, v FOV in the velodyne coordinates
        c_    - color value(HSV's Hue) corresponding to distance(m)

                 [x_1 , x_2 , .. ]
        xyz_v =  [y_1 , y_2 , .. ]   
                 [z_1 , z_2 , .. ]
                 [ 1  ,  1  , .. ]
        """
        xyz_v, c_ = self.velo_points_filter(points, v_fov, h_fov, filter_out_of_range=filter_out_of_range)

        """
        RT_ - rotation matrix & translation matrix
            ( velodyne coordinates -> camera coordinates )

                [r_11 , r_12 , r_13 , t_x ]
        RT_  =  [r_21 , r_22 , r_23 , t_y ]   
                [r_31 , r_32 , r_33 , t_z ]
        """
        RT_ = np.concatenate((R_vc, T_vc), axis=1)

        # convert velodyne coordinates(X_v, Y_v, Z_v) to camera coordinates(X_c, Y_c, Z_c)
        for i in range(xyz_v.shape[1]):
            xyz_v[:3, i] = np.matmul(RT_, xyz_v[:, i])

        """
        xyz_c - 3D velodyne points corresponding to h, v FOV in the camera coordinates
                 [x_1 , x_2 , .. ]
        xyz_c =  [y_1 , y_2 , .. ]   
                 [z_1 , z_2 , .. ]
        """
        xyz_c = np.delete(xyz_v, 3, axis=0)

        # convert camera coordinates(X_c, Y_c, Z_c) image(pixel) coordinates(x,y)
        for i in range(xyz_c.shape[1]):
            xyz_c[:, i] = np.matmul(P_, xyz_c[:, i])

        """
        xy_i - 3D velodyne points corresponding to h, v FOV in the image(pixel) coordinates before scale adjustment
        ans  - 3D velodyne points corresponding to h, v FOV in the image(pixel) coordinates
                 [s_1*x_1 , s_2*x_2 , .. ]
        xy_i =   [s_1*y_1 , s_2*y_2 , .. ]        ans =   [x_1 , x_2 , .. ]  
                 [  s_1   ,   s_2   , .. ]                [y_1 , y_2 , .. ]
        """
        xy_i = xyz_c[::] / xyz_c[::][2]
        ans = np.delete(xy_i, 2, axis=0)

        """
        width = 1242
        height = 375
        w_range = in_range_points(ans[0], width)
        h_range = in_range_points(ans[1], height)

        ans_x = ans[0][np.logical_and(w_range,h_range)][:,None].T
        ans_y = ans[1][np.logical_and(w_range,h_range)][:,None].T
        c_ = c_[np.logical_and(w_range,h_range)]

        ans = np.vstack((ans_x, ans_y))
        """

        return ans, c_

    def print_projection_cv2(self, points, color, image):
        """ project converted velodyne points into camera image """

        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        for i in range(points.shape[1]):
            cv2.circle(hsv_image, (np.int32(points[0][i]), np.int32(points[1][i])), 2, (int(color[i]), 255, 255), -1)

        return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    def print_projection_plt(self, points, color, image):
        """ project converted velodyne points into camera image """

        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        for i in range(points.shape[1]):
            cv2.circle(hsv_image, (np.int32(points[0][i]), np.int32(points[1][i])), 2, (int(color[i]), 255, 255), -1)

        return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)

    def write_tracklets_dataframe_to_csv(self,date,drives):
        for drive in drives:
            print('Processing drive: {}'.format(drive))
            cols = ['frame_number','tracklet_index','label', 'x1', 'y1', 'x2', 'y2','dist',
                    'sx','sy','sz','tx','ty','tz','rx','ry','rz',
                    'amt_borders_x','amt_borders_y','amt_borders_z',
                    'amt_occl_1','amt_occl_2','occ_1','occ_2','state','trunc',
                    'wx1','wy1','wz1','wx2','wy2','wz2','wx3','wy3','wz3','wx4','wy4','wz4',
                    'wx5','wy5','wz5','wx6','wy6','wz6','wx7','wy7','wz7','wx8','wy8','wz8',
                    'px1','py1','px2','py2','px3','py3','px4','py4',
                    'px5','py5','px6','py6','px7','py7','px8','py8',
                    'dist_seg0', 'dist_seg1', 'dist_seg2', 'dist_seg3',
                    'dist_seg4', 'dist_seg5', 'dist_seg6', 'dist_seg7',
                    'dist_seg8', 'dist_seg9', 'dist_seg10', 'dist_seg11',
                    'dist_seg12', 'dist_seg13', 'dist_seg14', 'dist_seg15']

            data_list = []
            dataset = self.load_dataset(date, drive)
            frames = len(list(dataset.velo))
            filepath = '../../data/{}/{}_drive_{}_sync/tracklet_labels.xml'.format(date, date, drive)
            tracklets = xmlParser.parseXML(filepath)

            # loop over tracklets
            for i, tracklet in enumerate(tracklets):
                # this part is inspired by kitti object development kit matlab code: computeBox3D
                h, w, l = tracklet.size
                # in velodyne coordinates around zero point and without orientation yet
                trackletBox = np.array([
                    [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2],
                    [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
                    [0.0, 0.0, 0.0, 0.0, h, h, h, h]
                ])
                # loop over all data in tracklet
                for translation, rotation, state, occlusion, truncation, amtOcclusion, amtBorders, absoluteFrameNumber in tracklet:
                    # determine if object is in the image; otherwise continue
#                    if truncation not in (xmlParser.TRUNC_IN_IMAGE, xmlParser.TRUNC_TRUNCATED):
#                        continue
                    # re-create 3D bounding box in velodyne coordinate system
                    yaw = rotation[2]  # other rotations are supposedly 0
                    assert np.abs(rotation[:2]).sum() == 0, 'object rotations other than yaw given!'
                    rotMat = np.array([
                        [np.cos(yaw), -np.sin(yaw), 0.0],
                        [np.sin(yaw), np.cos(yaw), 0.0],
                        [0.0, 0.0, 1.0]
                    ])

                    cornerPosInVelo = np.dot(rotMat, trackletBox).T + np.tile(translation, (8, 1))

                    projection_px = self.velo3d_2_camera2d_points(cornerPosInVelo, self.VFOV, self.HFOV, self.v2c_filepath, self.c2c_filepath,'02',filter_out_of_range=False)
                    (r, c) = np.shape(projection_px[0])
                    for ii in range(r):
                        for jj in range(c):
                            projection_px[0][ii,jj] = projection_px[0][ii,jj].astype(int)

                    x1 = projection_px[0][0,:].min().astype(int)
                    y1 = projection_px[0][1,:].min().astype(int)
                    x2 = projection_px[0][0,:].max().astype(int)
                    y2 = projection_px[0][1,:].max().astype(int)
                    x_min_world = cornerPosInVelo[:,0].min()
                    y_avg_world = cornerPosInVelo[:,1].mean()
                    world_dist = np.sqrt(x_min_world ** 2 + y_avg_world ** 2)


                    if i == 31 and False:
                        ans, c_ = kt.velo3d_2_camera2d_points(cornerPosInVelo, v_fov=(self.VFOV_MIN, self.VFOV_MAX),
                                                              h_fov=(self.HFOV_MIN, self.HFOV_MAX), \
                                                              vc_path=self.v2c_filepath, cc_path=self.c2c_filepath,
                                                              mode='02')

                        image = np.array(dataset.get_cam2(absoluteFrameNumber))
                        image = kt.print_projection_plt(points=ans, color=c_, image=image)
                        img = Image.fromarray(image)
                        img.show()

                    # find intersections of object edges with lidar segments (use bottom of box only)
                    dists = self.find_tracklet_distances_by_segment([
                        [cornerPosInVelo[0, 0], cornerPosInVelo[0, 1], cornerPosInVelo[1, 0], cornerPosInVelo[1, 1]],
                        [cornerPosInVelo[1, 0], cornerPosInVelo[1, 1], cornerPosInVelo[2, 0], cornerPosInVelo[2, 1]],
                        [cornerPosInVelo[2, 0], cornerPosInVelo[2, 1], cornerPosInVelo[3, 0], cornerPosInVelo[3, 1]],
                        [cornerPosInVelo[3, 0], cornerPosInVelo[3, 1], cornerPosInVelo[0, 0], cornerPosInVelo[0, 1]]
                    ])

                    if not (x2 < 0 or x1 > cal.cal['X_RESOLUTION'] or y2 < 0 or y1 > cal.cal['Y_RESOLUTION']\
                        or cornerPosInVelo[0, 0] < 0 or cornerPosInVelo[1, 0] < 0 or cornerPosInVelo[2, 0] < 0 or cornerPosInVelo[3, 0] < 0 \
                        or cornerPosInVelo[4, 0] < 0 or cornerPosInVelo[5, 0] < 0 or cornerPosInVelo[6, 0] < 0 or cornerPosInVelo[7, 0] < 0 ):
                        data_list.append([
                            int(absoluteFrameNumber), i, tracklet.objectType,  x1, y1, x2, y2, world_dist,
                            tracklet.size[0], tracklet.size[1], tracklet.size[2],
                            translation[0], translation[1], translation[2],
                            rotation[0], rotation[1], rotation[2],
                            int(amtBorders[0]), int(amtBorders[1]), int(amtBorders[2]),
                            int(amtOcclusion[0]), int(amtOcclusion[1]), int(occlusion[0]), int(occlusion[1]), int(state), int(truncation),
                            cornerPosInVelo[0, 0], cornerPosInVelo[0, 1], cornerPosInVelo[0, 2],
                            cornerPosInVelo[1, 0], cornerPosInVelo[1, 1], cornerPosInVelo[1, 2],
                            cornerPosInVelo[2, 0], cornerPosInVelo[2, 1], cornerPosInVelo[2, 2],
                            cornerPosInVelo[3, 0], cornerPosInVelo[3, 1], cornerPosInVelo[3, 2],
                            cornerPosInVelo[4, 0], cornerPosInVelo[4, 1], cornerPosInVelo[4, 2],
                            cornerPosInVelo[5, 0], cornerPosInVelo[5, 1], cornerPosInVelo[5, 2],
                            cornerPosInVelo[6, 0], cornerPosInVelo[6, 1], cornerPosInVelo[6, 2],
                            cornerPosInVelo[7, 0], cornerPosInVelo[7, 1], cornerPosInVelo[7, 2],
                            int(projection_px[0][0,0]), int(projection_px[0][1,0]), int(projection_px[0][0,1]), int(projection_px[0][1,1]),
                            int(projection_px[0][0,2]), int(projection_px[0][1,2]), int(projection_px[0][0,3]), int(projection_px[0][1,3]),
                            int(projection_px[0][0,4]), int(projection_px[0][1,4]), int(projection_px[0][0,5]), int(projection_px[0][1,5]),
                            int(projection_px[0][0,6]), int(projection_px[0][1,6]), int(projection_px[0][0,7]), int(projection_px[0][1,7]),
                            dists[0], dists[1], dists[2], dists[3], dists[4], dists[5], dists[6], dists[7],
                            dists[8], dists[9], dists[10], dists[11], dists[12], dists[13], dists[14], dists[15]
                        ])

                    a=1
            tracklet_df = pd.DataFrame(data_list, columns=cols)
            filepath = '../../data/{}_drive_{}_sync_complete_tracklets.csv'.format(date,drive)

            if not os.path.exists(filepath):
                tracklet_df.to_csv(filepath, index=False )
        return

    def test_crossing(self,lineA,lineB,x,y):
        x1 = lineA[0]; y1 = lineA[1]; x2 = lineA[2]; y2 = lineA[3]
        x3 = lineB[0]; y3 = lineB[1]; x4 = lineB[2]; y4 = lineB[3]

        maxAx = max(x1,x2)
        minAx = min(x1,x2)
        maxBx = max(x3,x4)
        minBx = min(x3,x4)
        maxAy = max(y1,y2)
        minAy = min(y1,y2)
        maxBy = max(y3,y4)
        minBy = min(y3,y4)
        if x >= minAx and x <= maxAx and x >= minBx and x <= maxBx and y >= minAy and y <= maxAy and y >= minBy and y <= maxBy:
            return True
        else:
            return False

    def find_intersections(self,lineA, lineB):
        x1 = lineA[0]; y1 = lineA[1]; x2 = lineA[2]; y2 = lineA[3]
        x3 = lineB[0]; y3 = lineB[1]; x4 = lineB[2]; y4 = lineB[3]

        # test for crossing
        if x1 == x2: # line A is vertical
            if x3 == x4: # line B is vertical
                return np.nan, np.nan # both lines are vertical (do not cross)
            else: # line B is not vertical
                mb = (float(y4) - float(y3))/(float(x4) - float(x3))
                y = mb * (x1 - x3) + y3
                x = x1
                if self.test_crossing(lineA,lineB,x,y):
                    return x, y # lineA is vertical but lineB is not and they cross
                else:
                    return np.nan, np.nan # lineA is vertical but lineB is not but they do not cross

        elif x3 == x4: # line B is vertical
            if x1 == x2: # line A is vertical
                return np.nan, np.nan # this is a redundant test both lines are vertical
            else: # line A is not vertical
                ma = (float(y2) - float(y1))/(float(x2) - float(x1))
                y = ma * (x3 - x1) + y1
                x = x3
                if self.test_crossing(lineA,lineB,x,y):
                    return x, y # lineB is vertical but lineA is not and they cross
                else:
                    return np.nan, np.nan  # lineB is vertical but lineA is not but they do not cross
        else: # neither line A nor line B is vertical
            ma = (float(y2) - float(y1)) / (float(x2) - float(x1))
            mb = (float(y4) - float(y3)) / (float(x4) - float(x3))
            if ma == mb: # both lines have the same slope
                return np.nan, np.nan # neither line is vertical but they have identical slopes so they don't cross
            else: # both lines do not have the same slope
                y = (-ma * y3 + ma * mb * (x3 - x1) + mb * y1) / (mb - ma)
                if mb == 0:
                    x = (y - y1) / ma + x1
                else:
                    x = (y - y3) / mb + x3
                if self.test_crossing(lineA,lineB,x,y):
                    return x, y # neither line is vertical and they cross
                else:
                    return np.nan, np.nan # neither line is vertical but they do not cross

    def find_tracklet_distances_by_segment(self,lines):
        dists = []
        #TODO - add display x,y pixel location of intersection
        xy = []
        for i in range(16):
            seg_dists = []
            for j, line in enumerate(lines):
                x_l,y_l = self.find_intersections(self.SEGMENTS_LEFT[i],line)
                x_r,y_r = self.find_intersections(self.SEGMENTS_RIGHT[i],line)
                if not (np.isnan(x_l) or np.isnan(x_r)):
                    x = (x_l + x_r)/2
                    y = (y_l + y_r)/2
                    seg_dists.append(math.sqrt(x * x + y * y))
                elif not np.isnan(x_l):
                    x = x_l
                    y = y_l
                    seg_dists.append(math.sqrt(x * x + y * y))
                elif not np.isnan(x_r):
                    x = x_r
                    y = y_r
                    seg_dists.append(math.sqrt(x * x + y * y))
                else:
                    pass

            if len(seg_dists) == 0:
                dists.append(0)
            else:
                dists.append(min(seg_dists))
        return dists


class M16Reading():
    def __init__(self, segment, x, y, color, distance):
        self.segment = segment
        self.x = x
        self.y = y
        self.color = color
        self.distance = distance



if __name__ == '__main__':
    date = '2011_09_26'
    drives = ['0001', '0002', '0005', '0009', '0015']
#    drives = ['0009']
    kt = Kitti_Tools()


    kt.write_tracklets_dataframe_to_csv(date, drives)

    # drive = '0001'
    drive = '0015'

    dataset = kt.load_dataset(date, drive)
    frames = len(list(dataset.velo))
    filepath = '../../data/{}/{}_drive_{}_sync/tracklet_labels.xml'.format(date, date, drive)

    tracklets = xmlParser.parseXML(filepath)

    tracklet_rects, tracklet_types = kt.load_tracklets_for_frames(frames,filepath)

    frame = 164

#    kt.display_frame_statistics(dataset, tracklet_rects, tracklet_types, frame)

    n_frames = len(list(dataset.velo))
    #frames = prepare_animation_frames(n_frames)
    #clip = ImageSequenceClip(frames, fps=5)
    #clip.write_gif('pcl_data.gif', fps=5)

    v2c_filepath = '../../data/2011_09_26/calib_velo_to_cam.txt'
    c2c_filepath = '../../data/2011_09_26/calib_cam_to_cam.txt'

    frame_number = 167
    image = np.array(dataset.get_cam2(frame_number))
    velo_pts_raw = dataset.get_velo(frame_number)
    ans, c_ = kt.velo3d_2_camera2d_points(velo_pts_raw, v_fov=(-24.9, 2.0), h_fov=(-45,45), \
                                   vc_path=v2c_filepath, cc_path=c2c_filepath, mode='02')

    image = kt.print_projection_plt(points=ans, color=c_, image=image)

    img = Image.fromarray(image)

    img.show()

    a = 1
    #
    # # subsample every Nth point
    # N = 1
    # pts = velo_pts_raw[::N]
    # # Filter x, y, z to in front of car
    # pts = pts[pts[:, 0] >= 0]
    # pts = pts[pts[:, 1] >= -5]
    # pts = pts[pts[:, 1] <= 5]
    #
    # # pts = pts[pts[:, 2] >= 0]
    #
    # fig = plt.figure(figsize=(12, 8))
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter3D(xs=pts[:, 0], ys=pts[:, 1], zs=pts[:, 2], s=0.5)
    # ax.set_xlim3d(pts[:, 0].min(), pts[:, 0].max())
    # # ax.set_xlim3d(0,25)
    # ax.set_ylim3d(pts[:, 1].min(), pts[:, 1].max())
    # # ax.set_ylim3d(-1, 1)
    # ax.set_zlim3d(pts[:, 2].min(), pts[:, 2].max())
    # # ax.set_zlim3d(-1.5, 0)
    #
    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # ax.set_zlabel('Z Label')
    #
    # plt.show()
    #
    # NUM_SEG = 16
    # SEG_WIDTH = 46.9 / NUM_SEG
    # SEG_WIDTH
    #
    #
    # # filter a slice of data where z coordinates lie between min/max vertical FOV
    # VFOV_MIN = -2
    # VFOV_MAX = 1
    # HFOV_MIN = -46.9 / 2
    # HFOV_MAX = 46.9 / 2
    # dataframes = []  # append dataframes from each frame number to create a dataset
    # for frame_number in tqdm(range(len(dataset.cam2_files))):
    #     image = np.array(dataset.get_cam2(frame_number))
    #     velo_pts_raw = dataset.get_velo(frame_number)
    #
    #     # velo_pts_raw_filtered = velo_pts_raw_filtered[velo_pts_raw_filtered[:, 2] <= VFOV_MAX]
    #     # velo_pts_raw_filtered = velo_pts_raw_filtered[velo_pts_raw_filtered[:, 2] >= VFOV_MIN]
    #     hfov_seg_min = HFOV_MIN
    #     hfov_seg_max = hfov_seg_min + SEG_WIDTH
    #     m16_readings = []
    #
    #     for seg in range(NUM_SEG):
    #         # image = np.array(dataset.get_cam2(frame_number))
    #         velo_pts_raw_filtered = velo_pts_raw.copy()
    #         velo_pts_raw_filtered, _ = kt.velo_points_filter(velo_pts_raw_filtered,
    #                                                       v_fov=(VFOV_MIN, VFOV_MAX),
    #                                                       h_fov=(hfov_seg_min, hfov_seg_max))
    #         if velo_pts_raw_filtered.any():
    #             velo_pts_raw_filtered = velo_pts_raw_filtered.T
    #             ans, c_ = kt.velo3d_2_camera2d_points(velo_pts_raw_filtered,
    #                                                v_fov=(VFOV_MIN, VFOV_MAX),
    #                                                h_fov=(HFOV_MIN, HFOV_MAX),
    #                                                vc_path=v2c_filepath,
    #                                                cc_path=c2c_filepath,
    #                                                mode='02')
    #             image = kt.print_projection_plt(points=ans, color=c_, image=image)
    #             hfov_seg_min = hfov_seg_max
    #             hfov_seg_max += SEG_WIDTH
    #             # use minimum value as lidar reading
    #             min_idx = np.argmin(velo_pts_raw_filtered[:, 0], axis=0)
    #             min_velo_pt = velo_pts_raw_filtered[min_idx, :].reshape(-1, 1).T
    #             min_camera_pt, _ = kt.velo3d_2_camera2d_points(min_velo_pt,
    #                                                         v_fov=(VFOV_MIN, VFOV_MAX),
    #                                                         h_fov=(HFOV_MIN, HFOV_MAX),
    #                                                         vc_path=v2c_filepath,
    #                                                         cc_path=c2c_filepath,
    #                                                         mode='02')
    #             min_dist = np.squeeze(min_velo_pt)[0]
    #             min_x, min_y = min_camera_pt
    #             cv2.circle(image, (int(min_x), int(min_y)), 10, (0, 255, 0), -1)
    #             cv2.putText(image, str(int(min_dist)), (int(min_x), int(min_y) - 10), 1, 1.5, (0, 255, 0), 2)
    #
    #             lidar_reading = M16Reading(segment=seg,
    #                                        x=min_x,
    #                                        y=min_y,
    #                                        color=c_,
    #                                        distance=min_dist)
    #         else:
    #             lidar_reading = M16Reading(segment=seg,
    #                                        x=None,
    #                                        y=None,
    #                                        color=None,
    #                                        distance=0)
    #             # plt.figure(figsize=(18,12))
    #         # plt.title('SEGMENT {}'.format(seg))
    #         # plt.imshow(image)
    #         # plt.show()
    #         m16_readings.append(lidar_reading)
    #     m16_readings_headers = [
    #         'frame',
    #         'unix_time',
    #         'elapsed_time',
    #         'fps',
    #         'segment',
    #         'distance',
    #         'amplitude',
    #         'flags'
    #     ]
    #     rows = []
    #     for m16_reading in m16_readings:
    #         row = {}
    #         row['frame'] = frame_number
    #         row['unix_time'] = -1
    #         row['elapsed_time'] = -1
    #         row['fps'] = -1
    #         row['segment'] = m16_reading.segment
    #         row['distance'] = m16_reading.distance
    #         row['amplitude'] = 200
    #         row['flags'] = -1
    #         rows.append(row)
    #
    #     m16_readings_df = pd.DataFrame(data=rows, columns=m16_readings_headers)
    #     dataframes.append(m16_readings_df)
    #
    # df_combined = pd.concat(dataframes)
    #
    # dataset.cam2_files[0]
    #
    # df_combined.to_csv('data/2011_09_26_drive_0015_sync_converted.csv', index=False)
    #
    # tracklet_rects[0][0]

