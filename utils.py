import numpy as np
import cv2
import glob
def birds_eye(image):
    img_size = (image.shape[1], image.shape[0])
    src = np.float32([[520, 460],[750, 460],
                      [1250, 720],[40, 720]])
    dst = np.float32([[0, 0], [1280, 0],
                     [1130, 720],[160, 720]])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(image, M, img_size)
    return warped, Minv

def apply_thresholds(image, l_thresh = (215,255), b_thresh = (145,200)):
    # Convert image to Lab and isolate the b channel for yellow lines
    b_channel = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)[:,:,2]

    # Threshold b channel
    b_thresh_min = b_thresh[0]
    b_thresh_max = b_thresh[1]
    b_binary = np.zeros_like(b_channel)
    b_binary[(b_channel >= b_thresh_min) & (b_channel <= b_thresh_max)] = 1

    # Convert image to LUV and isolate the L channel for white lines
    l_channel = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)[:,:,0]

    # Threshold L channel
    l_thresh_min = l_thresh[0]
    l_thresh_max = l_thresh[1]
    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel >= l_thresh_min) & (l_channel <= l_thresh_max)] = 1

    # Create combined binary image
    combined_binary = np.zeros_like(l_binary)
    combined_binary[(l_binary == 1) | (b_binary == 1)] = 1

    return combined_binary

def calibrate_camera():
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    images = glob.glob('camera_cal/calibration*.jpg')

    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        img_size = (img.shape[1], img.shape[0])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    return mtx, dist