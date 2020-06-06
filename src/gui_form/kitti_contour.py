import numpy as np
import wx
from matplotlib import pyplot as plt, cm
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.figure import Figure


class CanvasPanel3(wx.Panel):
    '''This is a custom container for the matplotlib coutour plot '''
    def __init__(self, parent, parent_data):
        wx.Panel.__init__(self, parent)
        plt.ion()
        self.figure = Figure()
        self.parent_data = parent_data
        self.parent = parent
        self.axes = self.figure.add_subplot(111)
        self.canvas = FigureCanvas(parent.m_panel_kitti_contours, -1, self.figure)

        fgSizer_canvas = wx.FlexGridSizer(1, 1, 0, 0)
        fgSizer_canvas.AddGrowableCol(0)
        fgSizer_canvas.AddGrowableRow(0)
        fgSizer_canvas.SetFlexibleDirection(wx.BOTH)
        fgSizer_canvas.SetNonFlexibleGrowMode(wx.FLEX_GROWMODE_SPECIFIED)
        fgSizer_canvas.Add(self.canvas, 1, wx.EXPAND | wx.ALL, 5)

        parent.m_panel_clusters.SetSizer(fgSizer_canvas)
        parent.m_panel_clusters.Layout()
        fgSizer_canvas.Fit(parent.m_panel_kitti_contours)

        plt.pause(0.1)


    def draw(self, clustered_df_frame, vmin, vmax, xlim, ylim):
        '''this method updates the contour plot'''

        self.axes.remove()
        self.axes = self.figure.add_subplot(111)
        X = clustered_df_frame['theta_prime']
        Y = clustered_df_frame['phi']
        Z = clustered_df_frame['rho']

        self.axes.tricontourf(-Y, X, Z, vmin=vmin, vmax=vmax, cmap=cm.coolwarm)
        surf = plt.cm.ScalarMappable(cmap=cm.coolwarm)
        surf.set_array(Z)
        surf.set_clim(float(vmin), float(vmax))
        self.figure.colorbar(surf, boundaries=np.linspace(vmin, vmax, 11))
        self.axes.grid()
#        self.axes.xlim(xlim[0],xlim[1])
#        self.axes.ylim(ylim[0],ylim[1])
        self.axes.set_title('Contour Map of kitti distances')