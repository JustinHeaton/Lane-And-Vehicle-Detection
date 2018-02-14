import tensorflow as tf
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageColor, ImageFont
import matplotlib.image as mpimg
from scipy.misc import imresize
from moviepy.editor import VideoFileClip
import numpy as np
import glob
import matplotlib.pyplot as plt
from collections import deque
from scipy.misc import imresize
import argparse
from utils import *

def filter_boxes(min_score, boxes, scores, classes):
    """Return boxes with a confidence >= `min_score`"""
    n = len(classes)
    idxs = []
    for i in range(n):
        if scores[i] >= min_score:
            idxs.append(i)

    filtered_boxes = boxes[idxs, ...]
    filtered_scores = scores[idxs, ...]
    filtered_classes = classes[idxs, ...]
    return filtered_boxes, filtered_scores, filtered_classes

def to_image_coords(boxes, height, width):
    """
    The original box coordinate output is normalized, i.e [0, 1].

    This converts it back to the original coordinate based on the image
    size.
    """
    box_coords = np.zeros_like(boxes)
    box_coords[:, 0] = boxes[:, 0] * height
    box_coords[:, 1] = boxes[:, 1] * width
    box_coords[:, 2] = boxes[:, 2] * height
    box_coords[:, 3] = boxes[:, 3] * width

    return box_coords

def draw_boxes(image, boxes, classes, scores, thickness=4):
    """Draw bounding boxes on the image"""
    draw = ImageDraw.Draw(image)
    for i in range(len(boxes)):
        bot, left, top, right = boxes[i, ...]
        height = abs(bot - top)
        width = right - left

        class_id = int(classes[i])
        color = COLOR_LIST[class_id]
        if classes[i] < len(categories)+1:
            #if height > width:
            draw.text((left+2, bot-11),categories[int(classes[i]-1)]+" "+str(int(scores[i]*100))+'%',(255,255,255))
            draw.line([(left, top), (left, bot), (right, bot), (right, top), (left, top)], width=thickness, fill=color)

def load_graph(graph_file):
    """Loads a frozen inference graph"""
    graph = tf.Graph()
    with graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(graph_file, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return graph



class Line:
    def __init__(self, queue_len = 10):
        # Was the line found in the previous frame?
        self.found = False

        # Remember x and y values of lanes in previous frame
        self.X = None
        self.Y = None

        # Store recent x intercepts for averaging across frames
        self.x_int = deque(maxlen=queue_len)
        self.top = deque(maxlen=queue_len)

        # Remember previous x intercept to compare against current one
        self.lastx_int = None
        self.last_top = None

        # Remember radius of curvature
        self.radii = deque(maxlen=queue_len)

        self.recent_fits = deque(maxlen=queue_len)

        # Store recent polynomial coefficients for averaging across frames
        self.fit0 = deque(maxlen=queue_len)
        self.fit1 = deque(maxlen=queue_len)
        self.fit2 = deque(maxlen=queue_len)
        self.fitx = None
        self.pts = []

        # Count the number of frames
        self.count = 0

    def found_search(self, x, y):
        '''
        This function is applied when the lane lines have been detected in the previous frame.
        It uses a sliding window to search for lane pixels in close proximity (+/- 25 pixels in the x direction)
        around the previous detected polynomial.
        '''
        xvals = []
        yvals = []
        if self.found == True:
            i = 720
            j = 630
            while j >= 0:
                yval = np.mean([i,j])
                xval = (np.mean(self.fit0))*yval**2 + (np.mean(self.fit1))*yval + (np.mean(self.fit2))
                x_idx = np.where((((xval - 25) < x)&(x < (xval + 25))&((y > j) & (y < i))))
                x_window, y_window = x[x_idx], y[x_idx]
                if np.sum(x_window) != 0:
                    np.append(xvals, x_window)
                    np.append(yvals, y_window)
                i -= 90
                j -= 90
        if np.sum(xvals) == 0:
            self.found = False # If no lane pixels were detected then perform blind search
        return xvals, yvals, self.found

    def blind_search(self, x, y, image):
        '''
        This function is applied in the first few frames and/or if the lane was not successfully detected
        in the previous frame. It uses a slinding window approach to detect peaks in a histogram of the
        binary thresholded image. Pixels in close proimity to the detected peaks are considered to belong
        to the lane lines.
        '''
        xvals = []
        yvals = []
        if self.found == False:
            i = 720
            j = 630
            histogram = np.sum(image[image.shape[0]//2:], axis=0)
            if self == Right:
                peak = np.argmax(histogram[image.shape[1]//2:]) + image.shape[1]//2
            else:
                peak = np.argmax(histogram[:image.shape[1]//2])
            while j >= 0:
                x_idx = np.where((((peak - 100) < x)&(x < (peak + 100))&((y > j) & (y < i))))
                x_window, y_window = x[x_idx], y[x_idx]
                if np.sum(x_window) != 0:
                    xvals.extend(x_window)
                    yvals.extend(y_window)
                if np.sum(x_window) > 100:
                    peak = np.int(np.mean(x_window))
                i -= 90
                j -= 90
        if np.sum(xvals) > 0:
            self.found = True
        else:
            yvals = self.Y
            xvals = self.X
        return xvals, yvals, self.found

    def radius_of_curvature(self, xvals, yvals, lane_width):
        ym_per_pix = 21./720 # meters per pixel in y dimension

        xm_per_pix = 3.7/lane_width # meteres per pixel in x dimension
        fit_cr = np.polyfit(yvals*ym_per_pix, xvals*xm_per_pix, 2)
        curverad = ((1 + (2*fit_cr[0]*np.max(yvals)*ym_per_pix + fit_cr[1])**2)**1.5) \
                                     /np.absolute(2*fit_cr[0])
        return curverad

    def sort_vals(self, xvals, yvals):
        sorted_index = np.argsort(yvals)
        sorted_yvals = yvals[sorted_index]
        sorted_xvals = xvals[sorted_index]
        return sorted_xvals, sorted_yvals

    def get_intercepts(self, polynomial):
        bottom = polynomial[0]*720**2 + polynomial[1]*720 + polynomial[2]
        top = polynomial[0]*0**2 + polynomial[1]*0 + polynomial[2]
        return bottom, top

def process_video(image):
    # Undistort the image
    undist = cv2.undistort(image, mtx, dist, None, mtx)
    # Warp undistorted image to birds eye view perspective
    warped, Minv = birds_eye(undist)
    # Apply thresholds to create binary image
    combined_binary = apply_thresholds(warped)
    # Identify all non zero pixels in the binary image
    x, y = np.nonzero(np.transpose(combined_binary))
    if Left.found == True: # Search for left lane pixels around previous polynomial
        leftx, lefty, Left.found = Left.found_search(x, y)

    if Right.found == True: # Search for right lane pixels around previous polynomial
        rightx, righty, Right.found = Right.found_search(x, y)

    if Right.found == False: # Perform blind search for right lane lines
        rightx, righty, Right.found = Right.blind_search(x, y, combined_binary)

    if Left.found == False: # Perform blind search for left lane lines
        leftx, lefty, Left.found = Left.blind_search(x, y, combined_binary)

    lefty = np.array(lefty).astype(np.float32)
    leftx = np.array(leftx).astype(np.float32)
    righty = np.array(righty).astype(np.float32)
    rightx = np.array(rightx).astype(np.float32)
    if np.sum(lefty) > 0:
        # Calculate left polynomial fit based on detected pixels
        left_fit = np.polyfit(lefty, leftx, 2)
        Left.recent_fits.append(left_fit)

    left_fit = np.mean(Left.recent_fits, axis=0)

    ploty = np.linspace(0, image.shape[0]-1, image.shape[0])
    # Fit polynomial to detected pixels
    left_fitx = [left_fit[0]*y**2 + left_fit[1]*y + left_fit[2] for y in ploty]
    Left.fitx = left_fitx

    if np.sum(righty) > 0:
        # Calculate right polynomial fit based on detected pixels
        right_fit = np.polyfit(righty, rightx, 2)
        Right.recent_fits.append(right_fit)

    right_fit = np.mean(Right.recent_fits, axis=0)

    # Fit polynomial to detected pixels
    right_fitx = [right_fit[0]*y**2 + right_fit[1]*y + right_fit[2] for y in ploty]
    Right.fitx = right_fitx

    lane_width = right_fitx[-1] - left_fitx[-1]

    # Compute radius of curvature for each lane in meters
    left_curverad = Left.radius_of_curvature(np.float32(left_fitx), np.float32(ploty), lane_width)
    right_curverad = Right.radius_of_curvature(np.float32(right_fitx), np.float32(ploty), lane_width)

    Left.radii.append(left_curverad)
    Right.radii.append(right_curverad)

    average_curverad = (np.mean(Left.radii) + np.mean(Right.radii))/2

    # Calculate the vehicle position relative to the center of the lane
    position = (right_fitx[-1]+left_fitx[-1])/2
    distance_from_center = abs((image.shape[1]/2 - position)*3.7/lane_width)

    warp_zero = np.zeros_like(combined_binary).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    pts_left = np.array([np.transpose(np.vstack([Left.fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([Right.fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    cv2.polylines(color_warp, np.int_([pts]), isClosed=False, color=(0,0,255), thickness = 40)
    cv2.fillPoly(color_warp, np.int_(pts), (34,255,34))
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))
    result = cv2.addWeighted(undist, 1, newwarp, 0.5, 0)

    # Print distance from center on video
    if position > 640:
        cv2.putText(result, 'Vehicle is {:.2f}m left of center'.format(distance_from_center), (100,80),
                 fontFace = 16, fontScale = 2, color=(255,255,255), thickness = 2)
    else:
        cv2.putText(result, 'Vehicle is {:.2f}m right of center'.format(distance_from_center), (100,80),
                 fontFace = 16, fontScale = 2, color=(255,255,255), thickness = 2)
    # Print radius of curvature on video
    cv2.putText(result, 'Radius of Curvature {}(m)'.format(int(average_curverad)), (120,140),
             fontFace = 16, fontScale = 2, color=(255,255,255), thickness = 2)
    return result

def pipeline(img):
    draw_img = Image.fromarray(img)
    boxes, scores, classes = sess.run([detection_boxes, detection_scores, detection_classes], feed_dict={image_tensor: np.expand_dims(img, 0)})
    # Remove unnecessary dimensions
    boxes = np.squeeze(boxes)
    scores = np.squeeze(scores)
    classes = np.squeeze(classes)

    confidence_cutoff = 0.3
    # Filter boxes with a confidence score less than `confidence_cutoff`
    boxes, scores, classes = filter_boxes(confidence_cutoff, boxes, scores, classes)

    # The current box coordinates are normalized to a range between 0 and 1.
    # This converts the coordinates actual location on the image.
    width, height = draw_img.size
    box_coords = to_image_coords(boxes, height, width)

    # Each class with be represented by a differently colored box
    draw_boxes(draw_img, box_coords, classes, scores)

    result = process_video(np.array(draw_img))
    return result
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str,
                        help='video or image',
                        default = 'video')
    parser.add_argument('input', type=str,
                        help='Enter filepath of the video or image you want to process',
                        default = 'project_video.mp4')
    parser.add_argument('output', type=str,
                        help='Enter name of output video/image you wish to write',
                        default = 'output.mp4')
    args = parser.parse_args()
    mtx, dist = calibrate_camera()
    Left = Line()
    Right = Line()

    SSD_GRAPH_FILE = './models/ssd_inception_v2_coco_2017_11_17/frozen_inference_graph.pb'

    categories = ['person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light',
                  'fire hydrant','','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow',
                  'elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee',
                  'skis','snowboard','sports ball']

    cmap = ImageColor.colormap
    COLOR_LIST = sorted([c for c in cmap.keys()])

    detection_graph = load_graph(SSD_GRAPH_FILE)

    with tf.Session(graph=detection_graph) as sess:
        image_tensor = sess.graph.get_tensor_by_name('image_tensor:0')
        detection_boxes = sess.graph.get_tensor_by_name('detection_boxes:0')
        detection_scores = sess.graph.get_tensor_by_name('detection_scores:0')
        detection_classes = sess.graph.get_tensor_by_name('detection_classes:0')
        if args.mode == 'video':
            clip = VideoFileClip(args.input)
            video_output = args.output
            output_clip = clip.fl_image(pipeline)
            output_clip.write_videofile(video_output, audio=False)
        else:
            image = mpimg.imread(args.input)
            mpimg.imsave(args.output, pipeline(image))
