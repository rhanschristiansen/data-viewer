import pykitti
import numpy as np
from source import parseTrackletXML as xmlParser
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
#% matplotlib notebook


class EDA_Util():
    def __init__(self):
        self.basedir = 'data'
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

        return


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
        f, ax3 = plt.subplots(3, 1, figsize=(5, 12))
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
    #    import pdb; pdb.set_trace()
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
        pts_3d_extend = np.hstack((pts_3d, np.ones((n, 1))))
        print(('pts_3d_extend shape: ', pts_3d_extend.shape))
        pts_2d = np.dot(pts_3d_extend, np.transpose(P))  # nx3
        pts_2d[:, 0] /= pts_2d[:, 2]
        pts_2d[:, 1] /= pts_2d[:, 2]
        return pts_2d[:, 0:2]

    ## Visualize velodyne points and distances in image plane
    #Taken from https: // github.com / windowsub0406 / KITTI_Tutorial / blob / master / velo2cam_projection_detail.ipynb
    #User Changsub Bae: https: // github.com / windowsub0406

    def depth_color(self, val, min_d=0, max_d=120):
        """
        print Color(HSV's H value) corresponding to distance(m)
        close distance = red , far distance = blue
        """
        np.clip(val, 0, max_d, out=val)  # max distance is 120m but usually not usual
        return (((val - min_d) / (max_d - min_d)) * 120).astype(np.uint8)

    def in_h_range_points(self, points, m, n, fov):
        """ extract horizontal in-range points """
        return np.logical_and(np.arctan2(n, m) > (-fov[1] * np.pi / 180), np.arctan2(n, m) < (-fov[0] * np.pi / 180))

    def in_v_range_points(self, points, m, n, fov):
        """ extract vertical in-range points """
        return np.logical_and(np.arctan2(n, m) < (fov[1] * np.pi / 180), np.arctan2(n, m) > (fov[0] * np.pi / 180))

    def fov_setting(self, points, x, y, z, dist, h_fov, v_fov):
        """ filter points based on h,v FOV  """

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

    def in_range_points(self, points, size):
        """ extract in-range points """
        return np.logical_and(points > 0, points < size)

    def velo_points_filter(self, points, v_fov, h_fov):
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

        x_lim = self.fov_setting(x, x, y, z, dist, h_fov, v_fov)[:, None]
        y_lim = self.fov_setting(y, x, y, z, dist, h_fov, v_fov)[:, None]
        z_lim = self.fov_setting(z, x, y, z, dist, h_fov, v_fov)[:, None]

        # Stack arrays in sequence horizontally
        xyz_ = np.hstack((x_lim, y_lim, z_lim))
        xyz_ = xyz_.T

        # stack (1,n) arrays filled with the number 1
        one_mat = np.full((1, xyz_.shape[1]), 1)
        xyz_ = np.concatenate((xyz_, one_mat), axis=0)

        # need dist info for points color
        dist_lim = self.fov_setting(dist, x, y, z, dist, h_fov, v_fov)
        color = self.depth_color(dist_lim, 0, 70)

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

    def velo3d_2_camera2d_points(self, points, v_fov, h_fov, vc_path, cc_path, mode='02'):
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
        xyz_v, c_ = self.velo_points_filter(points, v_fov, h_fov)

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



