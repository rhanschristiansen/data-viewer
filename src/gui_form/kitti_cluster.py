import math
import numpy as np
import wx
from matplotlib import pyplot as plt
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.figure import Figure


class CanvasPanel(wx.Panel):
    '''this class is a container for the matplotlib plot for the kitti data clusters '''
    def __init__(self, parent, parent_data):
        wx.Panel.__init__(self, parent)
        plt.ion()
        self.figure = Figure()
        self.parent_data = parent_data
        self.parent = parent
        self.axes = self.figure.add_subplot(111)
        self.canvas = FigureCanvas(parent.m_panel_clusters, -1, self.figure)

        fgSizer_canvas = wx.FlexGridSizer(1, 1, 0, 0)
        fgSizer_canvas.AddGrowableCol(0)
        fgSizer_canvas.AddGrowableRow(0)
        fgSizer_canvas.SetFlexibleDirection(wx.BOTH)
        fgSizer_canvas.SetNonFlexibleGrowMode(wx.FLEX_GROWMODE_SPECIFIED)
        fgSizer_canvas.Add(self.canvas, 1, wx.EXPAND | wx.ALL, 5)

        parent.m_panel_clusters.SetSizer(fgSizer_canvas)
        parent.m_panel_clusters.Layout()
        fgSizer_canvas.Fit(parent.m_panel_clusters)

        # assume the lidar is centered on angle 0 with a 45 degree aperture
        aperture = 45
        center = 0
        b = aperture / 2 + center
        m = -aperture / 16

        self.lidar_seg_edges_angles = []

        for i in range(17):
            self.lidar_seg_edges_angles.append(math.radians(b + m * i))

        lidar_edge_len = 140
        self.lidar_edge_ends_x = []
        self.lidar_edge_ends_y = []

        for i in range(len(self.lidar_seg_edges_angles)):
            self.lidar_edge_ends_x.append(math.cos(self.lidar_seg_edges_angles[i]) * lidar_edge_len)
            self.lidar_edge_ends_y.append(math.sin(self.lidar_seg_edges_angles[i]) * lidar_edge_len)
        plt.pause(0.1)


    def draw(self, clustered_segs_only):
        '''this method updates the cluster plot'''

        self.axes.remove()
        self.axes = self.figure.add_subplot(111)

        self.axes.scatter(clustered_segs_only.x, clustered_segs_only.y, s=1)
        self.axes.grid()
        self.axes.set_xlim([0, 150])
        self.axes.set_ylim([-50, 50])
        self.axes.set_title('M16 lidar detection zones with DBSCAN clustering')
        self.axes.legend()

        cluster = clustered_segs_only.groupby(['segment','cluster_label'])

        cluster_mean = cluster.mean() # find the centers in the x and y space

        self.axes.scatter(cluster_mean.x, cluster_mean.y, marker='x', s=20, c='k')

        for i, xs, ys in zip(cluster_mean.index, cluster_mean.x, cluster_mean.y):
            self.axes.text(xs + 2, ys + 2, str(i), fontsize=10)

        for i in range(len(self.lidar_edge_ends_x)):
            line, _ = self.axes.plot([0, self.lidar_edge_ends_x[i]], [0, self.lidar_edge_ends_y[i]], 'g', linewidth=0.5)

        if self.parent_data.show_ground_truth:

            color = tuple(np.array(self.parent_data.ground_truth_color, dtype=np.float32) / 255)
            for t in self.parent_data.gt_frame.itertuples():
                linex = np.array([t.wx1, t.wx2, t.wx3, t.wx4, t.wx1],
                                 dtype=np.float32)*self.parent_data.cal.cal['M_TO_FT']
                liney = np.array([t.wy1, t.wy2, t.wy3, t.wy4, t.wy1],
                                 dtype=np.float32)*self.parent_data.cal.cal['M_TO_FT']
                self.axes.plot(linex, liney, color=color , linewidth=2)
