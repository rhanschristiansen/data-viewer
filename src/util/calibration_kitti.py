'''
This file contains the calibration information for the lidar information from the kitti dataset
and the cameras used on the kitti collection system

Assumptions: the lidar detector is pointed down from the horizon at an angle of 6.75 degrees and pointed
directly ahead. This provides for a lidar field of view -22.5 deg to +22.5 degrees on the horizontal and
-3.0 degrees to -10.5 degrees on the vertical.

It is also assumed that the varifocal lens is set to a focal length of 4mm for a horizontal field of view
of 77.55 deg. This needs to be confirmed experimentally from the video data.

'''
import math

cal = {
    'Y_HORIZON'        : 175,  #256  353, 363 this is the center of optical flow in the y (vertical) direction
    'Y_RESOLUTION'     : 375,  #512
    'X_CENTER'         : 621,  # this is the center of optical flow in the x (horizontal) direction
    'X_RESOLUTION'     : 1242,  #1382
    'VFOV'             : math.radians(31.57),
    'HFOV'             : math.radians(92.00),
    'FOCAL_LENGTH'     : 599.69,  # in pixels  = (X_RESOLUTION / 2) / tan( HFOV / 2 )
    'HT_CAMERA'        : 5.41,  # 5.58
    'WIDTH_CAR'        : 5.41,  # cars are slightly smaller in europe ;)
    'LENGTH_CAR'       : 15.09,
    'EDGE_MARGIN'      : 0,
    'M_TO_FT'          : 3.28084,
    'VIDEO_LAG_FRAMES' : 0
}

cal['WL_RATIO'] =  cal['WIDTH_CAR'] / cal['LENGTH_CAR']

SEG_TO_PIXEL_LEFT = {
0: 311, 1: 349, 2: 387, 3: 425,
4: 463, 5: 501, 6: 539, 7: 577,
8: 615, 9: 652,  10: 690, 11: 728,
12: 766, 13: 804, 14: 842, 15: 880
}

SEG_TO_PIXEL_CENTER = {
0: 330, 1: 368, 2: 406, 3: 444,
4: 482, 5: 520, 6: 558, 7: 596,
8: 633, 9: 671, 10: 709, 11: 747,
12: 785, 13: 823, 14: 861, 15: 899
}

SEG_TO_PIXEL_RIGHT = {
0: 349, 1: 387, 2: 425, 3: 463,
4: 501, 5: 539, 6: 577, 7: 615,
8: 652, 9: 690, 10: 728, 11: 766,
12: 804, 13: 842, 14: 880, 15: 918
}

SEG_TO_ANGLE = {
0: -21.09375, 1: -18.28125, 2: -15.46875, 3: -12.65625,
4: -9.84375, 5: -7.03125, 6: -4.21875, 7: -1.40625,
8: 1.40625, 9: 4.21875, 10: 7.03125, 11: 9.84375,
12: 12.65625, 13: 15.46875, 14: 18.28125, 15: 21.09375
}

SEG_TO_ANGLE_LEFT = {
0: -22.5,  1: -19.6875,  2: -16.875,  3: -14.0625,
4: -11.25, 5: -8.4375,   6: -5.625,   7: -2.8125,
8: 0,      9:  2.8125,  10: 5.625,   11: 8.4375,
12: 11.25, 13: 14.0625, 14: 16.875,  15: 19.6875
}

SEG_TO_ANGLE_RIGHT = {
0:  -19.6875,  1:  -16.875,  2:  -14.0625,  3: -11.25,
4:   -8.4375,  5:   -5.625,  6:   -2.8125,  7:   0,
8:    2.8125,  9:    5.625, 10:    8.4375, 11:  11.25,
12:  14.0625, 13:   16.875, 14:   19.6875, 15:  22.5
}

SEG_TO_PIXEL_TOP = 187
SEG_TO_PIXEL_BOTTOM  = 276
