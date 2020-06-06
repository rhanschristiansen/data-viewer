import cv2


def draw_detzone(self, image, color):
    '''This draws the detection zones for the lidar'''
    y1 = self.cal.SEG_TO_PIXEL_TOP
    y2 = self.cal.SEG_TO_PIXEL_BOTTOM
    thk = 2
    # draw number of segments based on the dropdown box selection on the GUI
    self.seg_step = 16 // self.n_segs
    for i in range(self.n_segs):
        x = self.cal.SEG_TO_PIXEL_LEFT[ i *self.seg_step]
        cv2.line(image, (x, y1), (x, y2), color=color, thickness=thk)
        cv2.line(image, (x - 5, y1), (x + 5, y1), color=color, thickness=thk)
        cv2.line(image, (x - 5, y2), (x + 5, y2), color=color, thickness=thk)

    x = self.cal.SEG_TO_PIXEL_RIGHT[15]
    cv2.line(image, (x, y1), (x, y2), color=color, thickness=thk)
    cv2.line(image, (x - 5, y1), (x + 5, y1), color=color, thickness=thk)
    cv2.line(image, (x - 5, y2), (x + 5, y2), color=color, thickness=thk)

    return image


def draw_lidar_values(self, image, color, text_size=0.8, text_weight = 1):
    '''this draws the lidar values below the detection area'''
    lidar_vals = self.lidar_frame
    self.right_click_lidar_locations = []
    y = int(self.cal.SEG_TO_PIXEL_BOTTOM + text_size * 15)
    for seg in range(self.n_segs):
        # lidar_seg_vals = lidar_vals.loc[lidar_vals['segment']==seg].sort_values(by=['distance'], ascending=False)
        lidar_seg_vals = lidar_vals.loc[lidar_vals['segment'] == seg] # don't sort so that the indexes stay in order
        for i in range(len(lidar_seg_vals)):
            xp = self.cal.SEG_TO_PIXEL_LEFT[seg*self.seg_step]+5
            yp = int(y+text_size*15*i)
            self.right_click_lidar_locations.append([xp+5, yp+5])
            cv2.putText(image, '{0:0.0f}'.format(lidar_seg_vals.iloc[i, 3]),
                        (xp, yp), 1, text_size, self.lidar_value_color, text_weight)
    return image


def draw_kitti_points(self, image):
    '''This draws the individual kitti detections on the screen'''
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    if self.show_all_kitti_points:
        df = self.clustered_df_frame
    else:
        df = self.clustered_segs_only
    for i in range(len(df)):
        row = df.iloc[i, :]
        cv2.circle(hsv_image, (int(row['x_px_2']), int(row['y_px_2'])), 2, (int(row['color']), 255, 255), -1)

    image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    return image


def update_displayed_object_lists(self):
    '''A helper method to assemble the lists of objects to be displayed on the screen'''
    self.df_list = []
    self.colors_list = []
    self.types_list = []
    if self.show_ground_truth:
        self.df_list.append(self.gt_frame)
        self.colors_list.append(self.ground_truth_color[0:3])
        self.types_list.append('ground_truth')
    if self.show_video_detections:
        self.df_list.append(self.det_frame)
        self.colors_list.append(self.video_detections_color[0:3])
        self.types_list.append('video_detections')
    if self.show_associations:
        self.df_list.append(self.association_frame)
        self.colors_list.append(self.associations_color[0:3])
        self.types_list.append('associations')
    if self.show_lidar_detections:
        self.df_list.append(self.lidar_frame)
        self.colors_list.append(self.lidar_detections_color[0:3])
        self.types_list.append('lidar_detections')


def draw_bboxes(self, image, df_list, colors_list, types_list):
    ''' pass bounding box objects to this method in a list, the drawing color and object type are also passed in as lists
        the association objects are treated a bit differently than the others
    '''
    y = self.cal.SEG_TO_PIXEL_TOP - 30
    for i, (df, color, type) in enumerate(zip(df_list, colors_list, types_list)):

        # TODO - rewrite draaw_bboxes to separate when record ground truth is enabled

        # associations have two circles with a line connecting them (different than the others)
        if type == 'associations':
            for j in range(len(df['assignments'])):
                asmt = [df['assignments'][j][0], df['assignments'][j][1]]

                # obj1 is either the detector or the ground truth bounding box (depending on whether "use detector" is checked
                # obj2 is the lidar bounding box

                # display only if the total cost is <= max_cost
                if (not self.enable_max_cost) or (df['total_costs'][asmt[0], asmt[1]] <= self.max_cost):
                    if self.use_detector:
                        obj1 = self.det_frame.iloc[asmt[0], :]
                        circle_color = self.video_detections_color[0:3]
                    else:
                        obj1 = self.gt_frame.iloc[asmt[0], :]
                        circle_color = self.ground_truth_color[0:3]
                    obj1_pt1 = (int(obj1.x1), int(obj1.y1))
                    obj1_pt2 = (int(obj1.x2), int(obj1.y2))
                    cv2.rectangle(img=image, pt1=obj1_pt1, pt2=obj1_pt2, color=color, thickness=1)

                    # make the circle filled in if it has been right clicked during manual recording of ground truth
                    circle_thk = 2
                    if self.enable_record_gt and asmt[0] == self.right_click_selected_object:
                        circle_thk = 7

                    cv2.circle(image,obj1_pt1,7,circle_color, circle_thk)
                    show_txt = True
                    show_line = True
                    circle_thk = 2
                    manual_association_index = -1
                    if self.enable_record_gt and j in self.remove_association_idx_frame: # association index is in remove list
                        found = False
                        for k in range(len(self.manual_associations_gt)):  # look for it in the manual associations list
                            # if in the list, use the manual association lidar value instead of the calculated association lidar value
                            if asmt[0] == self.manual_associations_gt[k][0]:
                                asmt[1] = self.manual_associations_gt[k][1]
                                manual_association_index = k
                                found = True
                        if found:
                            if self.manual_associations_gt[k][4]:
                                lidar_color = (0, 255, 0)
                            else:
                                lidar_color = self.lidar_detections_color[0:3]
                            show_txt = True # show the distance
                            show_line = True # show a cyan line
                            line_color = [0,255,255] # cyan
                        else:
                            lidar_color = [0,0,0]
                            show_txt = False # don't show the distance
                            show_line = False  # don't show the line
                            line_color = color
                    # association index is not in the remove association list and is a good association
                    elif self.enable_record_gt and j in self.valid_association_idx_frame:
                        lidar_color = [0, 255, 0] # use a green dot on the lidar circle to show it has been validated
                        show_txt = True # show the distance
                        show_line = True # show the line
                        line_color = color # use the standard line color
                    else:
                        lidar_color = self.lidar_detections_color[0:3]
                        show_txt = True
                        show_line = True
                        line_color = color
                    if asmt[1] == -1:
                        obj2 = self.lidar_frame.iloc[asmt[1], :]
                        obj2_pt1 = (int(obj1.x1), int(obj1.y1))
                        obj2_pt2 = (int(obj1.x1), int(obj1.y1))
                        obj2_text_pt = (int(obj1.x1 - 10), int(obj1.y1) - 10)
                        distance = self.manual_associations_gt[k][3]
                        if self.manual_associations_gt[k][4]: # magneta before recording, green afterwards
                            lidar_color = (0,255,0) # green
                        else:
                            lidar_color = (255,0,255) # magenta
                    else:
                        obj2 = self.lidar_frame.iloc[asmt[1],:]
                        obj2_pt1 = (int(obj2.x1), int(obj2.y1))
                        obj2_pt2 = (int(obj2.x2), int(obj2.y2))
                        obj2_text_pt = (int(obj2.x1-10), int(obj2.y1)-10)
                        distance = obj2.distance
                        cv2.rectangle(img=image, pt1=obj2_pt1, pt2=obj2_pt2, color=color, thickness=1)
                        cv2.circle(image, obj2_pt1, 7, lidar_color, 2)

                    if show_txt:
                        cv2.putText(image, '{0:0.0f}'.format(distance), obj2_text_pt, 1, 1.5, lidar_color, 2)
                    if show_line:
                        cv2.line(image, obj1_pt1, obj2_pt1, line_color, 2)

                    # stack numbers into rows above the objects to prevent them from overwriting each other
                    if self.show_index_numbers:
                        if self.use_detector:
                            obj1 = self.det_frame.iloc[asmt[0], :]
                        else:
                            obj1 = self.gt_frame.iloc[asmt[0], :]
                        if asmt[1] == -1:
                            obj2 = obj1
                        else:
                            obj2 = self.lidar_frame.iloc[asmt[1], :]
                        x = int((obj1.x1 + obj2.x1) / 2)
                        y = int((obj1.y1 + obj2.y1) / 2)
                        if show_line:
                            cv2.putText(image, '{0:0.0f}'.format(j), (x-5, y-20*i), 1, 1.5, color, 2)

        else: # all of the non-association type bounding boxes go here
            for j in range(len(df)):
                pt1 = (int(df.iloc[j,:].x1),int(df.iloc[j,:].y1))
                pt2 = (int(df.iloc[j,:].x2),int(df.iloc[j,:].y2))
                cv2.rectangle(img=image,pt1=pt1,pt2=pt2,color=color,thickness=2)
                if type == 'ground_truth':
                    # display the distance above the ground truth bounding box
                    if self.show_associations:
                        y_add = -20
                    else:
                        y_add = 0
                    cv2.putText(image, '{0:0.0f}'.format(df.iloc[j, 10]),
                                (int(df.iloc[j, :].x1 + 10), int(df.iloc[j, :].y1 - 20 + y_add)), 1, 1.5, color, 2)
                if self.show_index_numbers:
                    if type == 'lidar_detections': # since there are so many lidar detections
                        for seg in range(self.n_segs):
                            vals = df.loc[df.segment == seg].sort_values('lidar_index')
                            x = int((self.cal.SEG_TO_PIXEL_LEFT[seg*self.seg_step] + self.cal.SEG_TO_PIXEL_RIGHT[seg*self.seg_step]) / 2)
#                           y = int(df.iloc[j, :].y1 - 10 - 25 * i)
                            if len(vals) > 0:
                                for k in range(len(vals)):
                                    ynew = y - 20*i - 20*k
                                    cv2.putText(image, '{0:0.0f}'.format(int(vals.iloc[k, 1])), (x, ynew), 1, 1.5, color, 2)
                    else:
                        cv2.putText(image, '{0:0.0f}'.format(int(df.iloc[j,1])), (int(df.iloc[j,:].x1+10), y-20*i), 1, 1.5, color, 2)

    return image


def draw_3d_bboxes(self, image, df_list, colors_list, types_list):
    for i, (df, color, type) in enumerate(zip(df_list, colors_list, types_list)):
        if type == '3d_ground_truth':
            for j in range(len(df)):
                ptx_bot = [int(df.iloc[j,:].px1), int(df.iloc[j,:].px2), int(df.iloc[j,:].px3), int(df.iloc[j,:].px4), int(df.iloc[j,:].px1)]
                pty_bot = [int(df.iloc[j,:].py1), int(df.iloc[j,:].py2), int(df.iloc[j,:].py3), int(df.iloc[j,:].py4), int(df.iloc[j,:].py1)]
                ptx_top = [int(df.iloc[j,:].px5), int(df.iloc[j,:].px6), int(df.iloc[j,:].px7), int(df.iloc[j,:].px8), int(df.iloc[j,:].px5)]
                pty_top = [int(df.iloc[j,:].py5), int(df.iloc[j,:].py6), int(df.iloc[j,:].py7), int(df.iloc[j,:].py8), int(df.iloc[j,:].py5)]
                for k in range(4):
                    cv2.line(image,(ptx_bot[k],pty_bot[k]),(ptx_bot[k+1],pty_bot[k+1]), color, 2)
                    cv2.line(image,(ptx_top[k],pty_top[k]),(ptx_top[k+1],pty_top[k+1]), color, 2)
                    cv2.line(image,(ptx_top[k],pty_top[k]),(ptx_bot[k],pty_bot[k]), color, 2)
    return image
