'''
This file contains the calibration information for the
M16 lidar detector - Logitech camera combined sensor
with the video resolution in 720P mode and the

'''
import math

# this is the main dictionary object that contains the calibration information
cal = {
    'Y_HORIZON'        : 364,  # 353, 363 this is the center of optical flow in the y (vertical) direction
    'Y_RESOLUTION'     : 720,
    'X_CENTER'         : 647,  # this is the center of optical flow in the x (horizontal) direction
    'X_RESOLUTION'     : 1280,
    'VFOV'             : math.radians(39.53),
    'HFOV'             : math.radians(66.05),
    'FOCAL_LENGTH'     : 984.5728,  # in pixels  = (X_RESOLUTION / 2) / tan( HFOV / 2 )
    'HT_CAMERA'        : 5.58,  # 5.58
    'WIDTH_CAR'        : 5.58,
    'LENGTH_CAR'       : 15.09,
    'EDGE_MARGIN'      : 0,
    'M_TO_FT'          : 3.28084,
    'VIDEO_LAG_FRAMES' : 0
}

cal['WL_RATIO'] =  cal['WIDTH_CAR'] / cal['LENGTH_CAR']

SEG_TO_PIXEL_LEFT = {
    0: 200, 1: 254, 2: 308, 3: 363,
    4: 417, 5: 471, 6: 525, 7: 579,
    8: 634, 9: 688, 10: 742, 11: 796,
    12: 850, 13: 904, 14: 959, 15: 1013
}

SEG_TO_PIXEL_CENTER = {
    0: 227, 1: 281, 2: 336, 3: 390,
    4: 444, 5: 498, 6: 552, 7: 607,
    8: 661, 9: 715, 10: 769, 11: 823,
    12: 877, 13: 932, 14: 986, 15: 1040
}

SEG_TO_PIXEL_RIGHT = {
    0: 254, 1: 308, 2: 363, 3: 417,
    4: 471, 5: 525, 6: 579, 7: 634,
    8: 688, 9: 742, 10: 796, 11: 850,
    12: 904, 13: 959, 14: 1013, 15: 1067
}

SEG_TO_ANGLE = {
    0: -21.673,  1: -18.886,  2: -16.048,  3: -13.262,
    4: -10.475,  5:  -7.689,  6:  -4.902,  7:  -2.064,
    8:   0.722,  9:   3.509, 10:   6.295, 11:   9.082,
    12: 11.868, 13:  14.706, 14:  17.493, 15:  20.279
}

SEG_TO_PIXEL_TOP = 384
SEG_TO_PIXEL_BOTTOM  = 517
