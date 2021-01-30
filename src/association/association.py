import numpy as np
#import src.association.munkres
import munkres

class Association:
    def __init__(self):
        pass

    # the evaluate cost function receives to arguments:
    # 1 - a dictionary called cost functions with the function method name as the key and the weight as the value
    #
    #     cost_functions = { costs.dist_between_centroids : 0.334,
    #                        costs.dist_lidar_to_y2estimate : 0.333,
    #                        costs.inverse_intersection_over_union : 0.333 }
    #
    # 2 - a dictionary contained in **kwargs with the keys containing the names of the two lists of objects
    #     to be associated (video_detections and lidar_detections) and the values are the lists of objects to be
    #     associated.
    #
    #    kwargs = {'video_detections' : video_detections, 'lidar_detections' : lidar_detections}
    #
    #
    # The evaluate cost function evaluates the costs using methods that are contained in the Costs Class in the
    # costs.py file.
    # The function is called like this:
    #
    #     a = Association()
    #     costs = a.evaluate_cost(cost_functions, **kwargs)
    #
    # each of the elements of the costs array represent the cost value between the
    # i-th video_detection and the j-th lidar_detection where
    # cost (i,j) = weight[0] * cost_function[0](i, j) + weight[1] * cost_function[1](i,j) + ... + weight[n] * cost_function[n](i,j)
    # for n cost functions and weights in the cost_functions dictionary

    def evaluate_cost(self, cost_functions, names, **kwargs):

        array_size = []
        for k,v in kwargs.items():
            array_size.append(len(v))
        if k != 'lidar_detections':
            array_size = [array_size[1], array_size[0]]

        cost = np.zeros((array_size[0], array_size[1]), np.float64 )
        cost_components = [None] * len(names)
        function_names = []

        for function_name, weight in cost_functions.items():
            cost_component = function_name(**kwargs)
            cost += weight * cost_component
            cost_components[names.index(function_name.__func__.__name__)] = cost_component
            function_names.append(function_name.__func__.__name__) # for debugging only

        return cost, cost_components

    def compute_munkres(self, cost):
        m = munkres.Munkres()
        assignments = m.compute(cost)
        return assignments


# this is a test of the Association class using the cost methods contained in the Costs class
if __name__ == '__main__':

    import src.detection.detection as video_det
    import src.lidar.lidar_detection as lidar_det
    import src.association.costs as costs

    costs = costs.Costs(data_type='kitti')

    vdet0 = video_det.Detection()
    vdet0.bbox = [412, 375, 486, 421]
    vdet1 = video_det.Detection()
    vdet1.bbox = [762, 374, 799, 408]
    vdet2 = video_det.Detection()
    vdet2.bbox = [913, 338, 1020, 375]
    vdet3 = video_det.Detection()
    vdet3.bbox = [708, 374, 739, 400]
    vdet4 = video_det.Detection()
    vdet4.bbox = [613, 361, 650, 384]
    vdet5 = video_det.Detection()
    vdet5.bbox = [562, 369, 600, 396]
    vdet6 = video_det.Detection()
    vdet6.bbox = [774, 378, 990, 502]
    vdet7 = video_det.Detection()
    vdet7.bbox = [893, 350, 954, 377]
    vdet8 = video_det.Detection()
    vdet8.bbox = [171, 360, 301, 416]

    video_detections = [vdet0, vdet1, vdet2, vdet3, vdet4, vdet5, vdet6, vdet7, vdet8]

    ldet0 = lidar_det.LIDAR_detection(840, 2, 38.3435516357, 0)
    ldet1 = lidar_det.LIDAR_detection(840, 11, 12.4829711914, 0)
    ldet2 = lidar_det.LIDAR_detection(840, 12, 12.714263915999998, 0)
    ldet3 = lidar_det.LIDAR_detection(840, 12, 36.3725891113, 0)
    ldet4 = lidar_det.LIDAR_detection(840, 13, 12.671356201199998, 0)
    ldet5 = lidar_det.LIDAR_detection(840, 15, 12.3006744385, 0)
    lidar_detections = [ldet0, ldet1, ldet2, ldet3, ldet4, ldet5]

    # enter the cost method names as keys in the dictionary and weights as their values
    # the returned costs array will have a number of rows equal to the number of video_detection objects
    # and a number of columns equal to the nubmer of lidar_detection objects.
    #
    # each of the elements of the array represent the cost value between the i-th video_detection and the j-th lidar_detection
    # cost (i,j) = weight[0] * cost_function[0](i, j) + weight[1] * cost_function[1](i,j) + ... + weight[n] * cost_function[n](i,j)
    # for n cost functions and weights in the cost_functions dictionary

    cost_functions = { costs.dist_between_centroids : 0.334,
                       costs.dist_lidar_to_y2estimate : 0.333,
                       costs.inverse_intersection_over_union : 0.333 }

    a = Association()

    # enter the video_detections and lidar_detections lists into the kwargs dictionary
    kwargs = {'video_detections' : video_detections, 'lidar_detections' : lidar_detections}

    # evaluate the costs array by passing the cost_functions dictionary and the kwargs dictionary to the evaluate_costs method
    costs = a.evaluate_cost(cost_functions, **kwargs)

    b = 1