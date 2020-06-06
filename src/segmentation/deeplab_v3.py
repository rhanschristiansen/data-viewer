import os
import tarfile
import numpy as np
from PIL import Image
import cv2

import tensorflow as tf


def create_pascal_label_colormap():
    """Creates a label colormap used in PASCAL VOC segmentation benchmark.

    Returns:
      A Colormap for visualizing segmentation results.
    """
    colormap = np.zeros((256, 3), dtype=int)
    ind = np.arange(256, dtype=int)

    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((ind >> channel) & 1) << shift
        ind >>= 3

    return colormap


def label_to_color_image(label):
    """Adds color defined by the dataset colormap to the label.

    Args:
      label: A 2D array with integer type, storing the segmentation label.

    Returns:
      result: A 2D array with floating type. The element of the array
        is the color indexed by the corresponding element in the input label
        to the PASCAL color map.

    Raises:
      ValueError: If label is not of rank 2 or its value is larger than color
        map maximum entry.
    """
    if label.ndim != 2:
        raise ValueError('Expect 2-D input label')

    colormap = create_pascal_label_colormap()

    if np.max(label) >= len(colormap):
        raise ValueError('label value too large.')

    return colormap[label]


class DeepLabV3(object):
    """Class to load deeplab v3 model and run inference."""

    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    INPUT_SIZE = 513
    FROZEN_GRAPH_NAME = 'frozen_inference_graph'

    LABEL_NAMES = np.asarray([
        'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
        'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
        'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
    ])

    FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
    FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)

    def __init__(self):
        """Creates and loads pretrained deeplab model."""
        THIS_DIR = os.path.dirname(__file__)
        self.model_name = 'mobilenetv2_coco_voctrainaug'  # MobileNetV2 based model
        # self.model_name = 'xception_coco_voctrainaug'  # XCeption based model
        self.model_dir = '{}/deeplab_models/{}'.format(THIS_DIR, self.model_name)
        self.tarball_name = 'deeplab_model.tar.gz'
        self.tarball_path = os.path.join(self.model_dir, self.tarball_name)
        self.graph = tf.Graph()

        graph_def = None
        # Extract frozen graph from tar archive.
        tar_file = tarfile.open(self.tarball_path)
        for tar_info in tar_file.getmembers():
            if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
                file_handle = tar_file.extractfile(tar_info)
                graph_def = tf.GraphDef.FromString(file_handle.read())
                break

        tar_file.close()

        if graph_def is None:
            raise RuntimeError('Cannot find inference graph in tar archive.')

        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')

        self.sess = tf.Session(graph=self.graph)

    def predict(self, image):
        """Runs inference on a single image.
    
        Args:
          image: A PIL.Image object, raw input image.
    
        Returns:
          resized_image: RGB image resized from original input image.
          seg_map: Segmentation map of `resized_image`.
        """
        try:
            width, height = image.size
        except TypeError as e:
            height, width = image.shape[:2]
        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        try:
            resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
        except Exception as e:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            resized_image = cv2.resize(image, target_size)
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
        seg_map = batch_seg_map[0]
        return resized_image, seg_map


def save_segmentation_maps_from_video(video_filename, output_folder, start_frame, end_frame):
    from tqdm import tqdm
    segmenter = DeepLabV3()
    vc = cv2.VideoCapture()
    vc.open(video_filename)
    for current_frame_idx in tqdm(range(end_frame + 1)):
        _, img = vc.read()
        if img is None:
            print("frame is None. Exiting loop.")
            break
        if current_frame_idx < start_frame:  # hacky way to fast forward to the right frame
            continue
        resized_img, seg_map = segmenter.predict(img)
        seg_image = label_to_color_image(seg_map).astype(np.uint8)
        seg_image = cv2.resize(seg_image, (img.shape[1], img.shape[0]))
        output_filename = "{:08d}.png".format(current_frame_idx)
        output_filepath = os.path.join(output_folder, output_filename)
        cv2.imwrite(output_filepath, seg_image)


def run_segmentation_on_video(video_filename, start_frame):
    segmenter = DeepLabV3()
    vc = cv2.VideoCapture()
    vc.open(video_filename)
    current_frame_idx = 0
    while True:
        _, img = vc.read()
        current_frame_idx += 1
        if current_frame_idx <= start_frame:
            continue
        resized_img, seg_map = segmenter.predict(img)
        seg_image = label_to_color_image(seg_map).astype(np.uint8)
        seg_image = cv2.resize(seg_image, (img.shape[1], img.shape[0]))
        seg_image_vis = seg_image.copy()
        alpha = 0.7
        seg_overlay = cv2.addWeighted(seg_image_vis, alpha, img, 1 - alpha, 0)
        cv2.imshow("original", img)
        cv2.imshow("segmentation", seg_image)
        cv2.imshow("segmentation overlay", seg_overlay)

        key = cv2.waitKey(1)
        if key == ord('q') or key & 0xFF == 27:  # press q or esc
            break
        if key == ord('p'):  # pause video
            while True:
                cv2.imshow("original", img)
                cv2.imshow("segmentation", seg_image)
                cv2.imshow("segmentation overlay", seg_overlay)
                key = cv2.waitKey(1)
                if key == ord('p'):
                    break
    cv2.destroyAllWindows()


if __name__ == '__main__':

    THIS_DIR = os.path.dirname(__file__)
    video_filename = os.path.join(THIS_DIR, '../../../data/2018-09-18/0002.avi')
    output_folder = os.path.join(THIS_DIR,
                                 '../../../data/2018-09-18/2018-09-18_drive_0002_sync/image_02/data/segmentation_masks')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    start_frame = 650
    end_frame = 6008
    run_segmentation_on_video(video_filename=video_filename, start_frame=start_frame)
    # save_segmentation_maps_from_video(video_filename=video_filename,
    #                                  output_folder=output_folder,
    #                                  start_frame=start_frame,
    #                                  end_frame=end_frame)
