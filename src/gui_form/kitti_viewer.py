import pandas as pd
import wx

from src.gui_form.kitti_cluster import CanvasPanel
from src.gui_form.frame import Frame_Kitti_Viewer
from src.gui_form.kitti_contour import CanvasPanel3
from src.gui_form.kitti_histogram import CanvasPanel2


class MyFrame_Kitti_Viewer(Frame_Kitti_Viewer):
    '''This class is derived from the Frame_Kitti_Viewer class that is built by the wxFormBuilder application
    the Frame_Kitti_Viewer class is in the file frame.py that is auto generated by wxFormBuilder
    This class provides the functionality for the child window that pops up when the View Kitti Data button is pressed
    '''
    def __init__(self, parent, parent_data = None):
        Frame_Kitti_Viewer.__init__(self, parent=wx.GetApp().TopWindow)
        self.Bind(wx.EVT_PAINT, self.onPaint)
        self.parent = parent
        self.parent_data = parent_data
        self.frame_number = -1
        self.min_grid_rows = 100
        self.clustered_df_frame = None
        self.clustered_segs_only = None
        self.clustered_df_display = None
        self.canvas = CanvasPanel(self, parent_data)
        # these variables are used to improve performance by eliminating the need to refresh everything
        # every time the onPaint method is fired.
        self.isOld = {
            'show_clusters' : True,
            'show_segments' : True,
            'histogram'     : True
        }

        # these are the options for the dropdown boxes in the cluster view scatter plot
        self.show_clusters_base_options = []
        self.show_clusters_options = self.show_clusters_base_options # will be updated in update_grids()
        self.show_clusters_choice_idx = 0
        self.cluster_labels_frame = []
        self.show_segments_options = ['All','0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15']
        self.show_segments_choice_idx = 0
        self.m_choice_show_segments.SetItems(self.show_segments_options)
        self.m_choice_show_segments.Select(self.show_segments_choice_idx)

        # these are the histogram settings and options
        self.clustered_df_hist = None
        self.canvas2 = CanvasPanel2(self, parent_data)

        self.hist_objects_choice_idx = 0
        self.hist_objects_base_options = ['None']
        self.hist_objects_options = self.hist_objects_base_options # will be updated in update_hist
        self.m_choice_hist_object.SetItems(self.hist_objects_options)
        self.m_choice_hist_object.Select(self.hist_objects_choice_idx)

        self.hist_segments_choice_idx = 0
        self.hist_segments_base_options = ['None','All','0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15']
        self.hist_segments_options = self.hist_segments_base_options # will be updated in update_hist
        self.m_choice_hist_segment.SetItems(self.hist_segments_options)
        self.m_choice_hist_segment.Select(self.hist_segments_choice_idx)
        self.hist_object_list = []
        self.show_hist_region = False

        # these are the contour plot settings and options
        self.clustered_df_contour = None
        self.canvas3 = CanvasPanel3(self, parent_data)

        self.contour_objects_choice_idx = 0
        self.contour_objects_base_options = ['None']
        self.contour_objects_options = self.contour_objects_base_options  # will be updated in update_hist
        self.m_choice_contour_object.SetItems(self.contour_objects_options)
        self.m_choice_contour_object.Select(self.contour_objects_choice_idx)

        self.contour_segments_choice_idx = 0
        self.contour_segments_base_options = ['None', 'All', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11',
                                           '12', '13', '14', '15']
        self.contour_segments_options = self.contour_segments_base_options  # will be updated in update_hist
        self.m_choice_contour_segment.SetItems(self.contour_segments_options)
        self.m_choice_contour_segment.Select(self.contour_segments_choice_idx)
        self.contour_object_list = []
        self.show_contour_region = False

    def update_show_clusters_control(self):
        '''update the controls that are in the clusters display window'''

        # update the cluster selector control
        cluster = self.clustered_df_frame.groupby('cluster_label')
        cluster_rho_min = cluster.rho.min()
        a = pd.Series(range(len(cluster_rho_min)), index=range(len(cluster_rho_min)), name='cluster_label')
        b = pd.Series(cluster_rho_min, index=range(len(cluster_rho_min)), name='cluster_rho_min')
        cluster_rho_min = pd.concat([a, b], axis=1)
        cluster_rho_min.sort_values(['cluster_rho_min'], inplace=True)
        self.cluster_labels_frame = cluster_rho_min.cluster_label
        self.show_clusters_options = []
        self.show_clusters_options = self.show_clusters_base_options.copy()
        for label in self.cluster_labels_frame:
            self.show_clusters_options.append(str(label))

        self.m_choice_show_clusters.SetItems(self.show_clusters_options)
        self.m_choice_show_clusters.Select(self.show_clusters_choice_idx)
        self.isOld['show_clusters'] = False

    def update_grids(self):
        '''this method updates the information in the clusters grid that shows the values'''

        # update controls if needed:
        if self.isOld['show_clusters']:
            self.update_show_clusters_control()

        # filter dataframe based on Clusters and Segments Selection
        # TODO - fix this to account for cluster by segment methodology
        self.clustered_df_display = self.clustered_df_frame.loc[self.clustered_df_frame['cluster_label'] == int(self.show_clusters_options[self.show_clusters_choice_idx])]
        if not self.show_segments_choice_idx == 0:
            self.clustered_df_display = self.clustered_df_display.loc[self.clustered_df_display['segment'] == int(self.show_segments_options[self.show_segments_choice_idx])]

        # update the Clusters grid
        dfs = [self.clustered_df_display]
        grids = [self.m_grid_kitti_data]
        colors = [[255,255,255,255]]
        for df, grid, color, in zip(dfs, grids, colors):
            grid.ClearGrid()
            # resize the grids for the new number of readings
            (r, c) = df.shape
            r_df = r
            r = max(r, self.min_grid_rows)
            r_old = grid.GetNumberRows()
            c_old = grid.GetNumberCols()
            if r_old < r:
                grid.AppendRows(r - r_old)
            else:
                grid.DeleteRows(numRows=r_old - r)
            if c_old < c:
                grid.AppendCols(c - c_old)
            else:
                grid.DeleteCols(numCols=c_old - c)
            cols = list(df.columns.values)
            # update the column labels
            for i, col in enumerate(cols):
                grid.SetColLabelValue(i, col)
            # update the values and add the color index to the frame column
            for i in range(r_df):
                for j in range(c):
                    val = str(df.iloc[i, j])
                    grid.SetCellValue(i, j, val)
                    if j == 0:
                        grid.SetCellBackgroundColour(i, j, color)
            # clear the back ground color below the area of the data
            for i in range(r_df, self.min_grid_rows):
                grid.SetCellBackgroundColour(i, 0, (255, 255, 255, 255))

    def update_graphs(self):
        del self.canvas
        self.canvas = CanvasPanel(self, self.parent_data)
        self.canvas.draw(self.clustered_segs_only)
        self.m_panel_clusters.Refresh()

    def update_hist_objects_options(self):
        '''method to update the options in the histogram plot'''
        dets = self.parent_data.det_frame
        gts = self.parent_data.gt_frame
        self.hist_objects_options = self.hist_objects_base_options.copy()

        self.hist_object_list = []

        for i in range(len(dets)):
            det_index = dets.iloc[i,:].detection_index
            self.hist_objects_options.append('det-{0:0.0f}'.format(det_index))
            self.hist_object_list.append(['det',det_index])

        for i in range(len(gts)):
            gt_index = gts.iloc[i,:].gt_index
            self.hist_objects_options.append('gt-{0:0.0f}'.format(gt_index))
            self.hist_object_list.append(['gt',gt_index])

        self.m_choice_hist_object.SetItems(self.hist_objects_options)
        self.m_choice_hist_object.Select(self.hist_objects_choice_idx)

    def update_hist(self):
        '''called by the onPaint method to update the histogram plot'''
        del self.canvas2
        self.canvas2 = CanvasPanel2(self, self.parent_data)

        if self.hist_segments_choice_idx == 0:
            self.df_hist = self.clustered_df_frame.copy()
        else:
            self.df_hist = self.clustered_segs_only.copy()

            if not self.hist_segments_choice_idx == 1:
                seg = self.hist_segments_choice_idx - 2
                self.df_hist = self.df_hist[self.df_hist['segment'] == seg]

        if not self.hist_objects_choice_idx == 0:
            [obj_type,obj_num] = self.hist_object_list[self.hist_objects_choice_idx - 1]
            if obj_type == 'det':
                df = self.parent_data.det_frame
                result = df[df['detection_index'] == obj_num][['x1', 'y1', 'x2', 'y2']]
            else:
                df = self.parent_data.gt_frame
                result = df[df['gt_index'] == obj_num][['x1', 'y1', 'x2', 'y2']]

            x1 = int(result['x1'])
            y1 = int(result['y1'])
            x2 = int(result['x2'])
            y2 = int(result['y2'])

            self.df_hist = self.df_hist[self.df_hist['x_px_2'] >= x1]
            self.df_hist = self.df_hist[self.df_hist['x_px_2'] <= x2]
            self.df_hist = self.df_hist[self.df_hist['y_px_2'] >= y1]
            self.df_hist = self.df_hist[self.df_hist['y_px_2'] <= y2]

        vminstr = self.m_textCtrl_hist_vmin.GetValue()
        try:
            vmin = float(vminstr)
            self.df_hist = self.df_hist[self.df_hist['rho'] >= vmin]
        except:
            vmin = self.df_hist.rho.min()

        vmaxstr = self.m_textCtrl_hist_vmax.GetValue()
        try:
            vmax = float(vmaxstr)
            self.df_hist = self.df_hist[self.df_hist['rho'] <= vmax]
        except:
            vmax = self.df_hist.rho.max()


        self.canvas2.draw(self.df_hist)
        self.m_panel_kitti_histograms.Refresh()

    def update_contour_objects_options(self):
        '''used to update the contour plot options'''
        dets = self.parent_data.det_frame
        gts = self.parent_data.gt_frame
        self.contour_objects_options = self.contour_objects_base_options.copy()

        self.contour_object_list = []

        for i in range(len(dets)):
            det_index = dets.iloc[i,:].detection_index
            self.contour_objects_options.append('det-{0:0.0f}'.format(det_index))
            self.contour_object_list.append(['det',det_index])

        for i in range(len(gts)):
            gt_index = gts.iloc[i,:].gt_index
            self.contour_objects_options.append('gt-{0:0.0f}'.format(gt_index))
            self.contour_object_list.append(['gt',gt_index])

        self.m_choice_contour_object.SetItems(self.contour_objects_options)
        self.m_choice_contour_object.Select(self.contour_objects_choice_idx)

    def update_contour(self):
        '''called by the opPaint method to update the contour plot'''
        del self.canvas3
        self.canvas3 = CanvasPanel3(self, self.parent_data)

        if self.contour_segments_choice_idx == 0:
            self.df_contour = self.clustered_df_frame.copy()
        else:
            self.df_contour = self.clustered_segs_only.copy()

            if not self.contour_segments_choice_idx == 1:
                seg = self.contour_segments_choice_idx - 2
                self.df_contour = self.df_contour[self.df_contour['segment'] == seg]

        if not self.contour_objects_choice_idx == 0:
            [obj_type,obj_num] = self.contour_object_list[self.contour_objects_choice_idx - 1]
            if obj_type == 'det':
                df = self.parent_data.det_frame
                result = df[df['detection_index'] == obj_num][['x1', 'y1', 'x2', 'y2']]
            else:
                df = self.parent_data.gt_frame
                result = df[df['gt_index'] == obj_num][['x1', 'y1', 'x2', 'y2']]

            x1 = int(result['x1'])
            y1 = int(result['y1'])
            x2 = int(result['x2'])
            y2 = int(result['y2'])


            self.df_contour = self.df_contour[self.df_contour['x_px_2'] >= x1]
            self.df_contour = self.df_contour[self.df_contour['x_px_2'] <= x2]
            self.df_contour = self.df_contour[self.df_contour['y_px_2'] >= y1]
            self.df_contour = self.df_contour[self.df_contour['y_px_2'] <= y2]

        vminstr = self.m_textCtrl_contour_vmin.GetValue()
        try:
            vmin = float(vminstr)
            self.df_contour = self.df_contour[self.df_contour['rho'] >= vmin]
        except:
            vmin = self.df_contour.rho.min()

        vmaxstr = self.m_textCtrl_contour_vmax.GetValue()
        try:
            vmax = float(vmaxstr)
            self.df_contour = self.df_contour[self.df_contour['rho'] <= vmax]
        except:
            vmax = self.df_contour.rho.max()

        #TODO - set to limits of region that is selected
        xlim = [self.df_contour.x.min(),self.df_contour.x.max()]
        ylim = [self.df_contour.y.min(),self.df_contour.y.max()]

        self.canvas3.draw(self.df_contour, vmin, vmax, xlim, ylim)
        self.m_panel_kitti_contours.Refresh()

    def onPaint(self, evt):
        if self.parent_data.current_frame != self.frame_number or self.parent_data.isOld['clusters']:
            self.isOld['show_clusters'] = True
            self.frame_number = self.parent_data.current_frame
            self.clustered_df_frame, self.clustered_segs_only = self.parent_data.at.get_kitti_frame(self.parent_data.current_frame)
            self.update_hist_objects_options()
            self.update_contour_objects_options()
        self.update_grids()
        self.update_graphs()
        self.update_hist()
        self.update_contour()

    # these are the event handlers for the kitti data viewer window
    def process_show_clusters( self, event ):
        self.show_clusters_choice_idx = event.Selection
        self.Refresh()

    def process_show_segments( self, event ):
        self.show_segments_choice_idx = event.Selection
        self.Refresh()

    def process_hist_object( self, event ):
        self.hist_objects_choice_idx = event.Selection
        self.Refresh()

    def process_hist_segment( self, event ):
        self.hist_segments_choice_idx = event.Selection
        self.Refresh()

    def process_show_hist_region( self, event ):
        self.show_hist_region = (event.Selection == 1)
        self.Refresh()

    def process_contour_object( self, event ):
        self.contour_objects_choice_idx = event.Selection
        self.Refresh()

    def process_contour_segment( self, event ):
        self.contour_segments_choice_idx = event.Selection
        self.Refresh()

    def process_show_contour_region( self, event ):
        self.show_contour_region = (event.Selection == 1)
        self.Refresh()