import wx
from matplotlib import pyplot as plt
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.figure import Figure


class CanvasPanel2(wx.Panel):
    '''this is a customm container for the matplotlib histogram plot'''
    def __init__(self, parent, parent_data):
        wx.Panel.__init__(self, parent)
        plt.ion()
        self.figure = Figure()
        self.parent_data = parent_data
        self.parent = parent
        self.axes = self.figure.add_subplot(111)
        self.canvas = FigureCanvas(parent.m_panel_kitti_histograms, -1, self.figure)

        fgSizer_canvas = wx.FlexGridSizer(1, 1, 0, 0)
        fgSizer_canvas.AddGrowableCol(0)
        fgSizer_canvas.AddGrowableRow(0)
        fgSizer_canvas.SetFlexibleDirection(wx.BOTH)
        fgSizer_canvas.SetNonFlexibleGrowMode(wx.FLEX_GROWMODE_SPECIFIED)
        fgSizer_canvas.Add(self.canvas, 1, wx.EXPAND | wx.ALL, 5)

        parent.m_panel_clusters.SetSizer(fgSizer_canvas)
        parent.m_panel_clusters.Layout()
        fgSizer_canvas.Fit(parent.m_panel_kitti_histograms)

        plt.pause(0.1)

    def draw(self, clustered_df_frame):
        '''this method updates the histogram'''

        self.axes.remove()
        self.axes = self.figure.add_subplot(111)

        self.axes.hist(clustered_df_frame.rho)
        self.axes.grid()
        self.axes.set_title('Histogram of kitti distances')
        self.axes.legend()