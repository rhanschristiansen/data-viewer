import numpy as np


class Detection(object):
    def __init__(self):
        self.img = None
        self.bbox = None
        self.XZ = None
        self.fvec = np.array([1, 2, 3, 4], np.float64)  # init in case not needed
        self.frame_id = None
        self.num_misses = 0
        self.has_match = False
        self.num_matches = 0
        self.matches = []
        # added by sh
        self.score = 0
        self.det_class = None
        self.area = 0
        self.is_valid = False
        self.dist_est_y2 = 0.0
        self.intersecting_segs = []
        self.distance_estimates_by_segs = []


    def as_numpy_array(self):
        return np.array([np.hstack((self.frame_id, self.bbox, self.det_class, self.score, self.fvec))])

    @staticmethod
    def det_from_numpy_array(np_array):
        d = Detection()
        d.frame_id = np_array[0]
        d.bbox = np_array[1:5]
        d.fvec = np_array[5:]

        return d
