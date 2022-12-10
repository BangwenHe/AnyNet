import glob
import os

import numpy as np
import cv2 as cv2


def load_stereo_coefficients(path):
    """ Loads stereo matrix coefficients. """
    # FILE_STORAGE_READ
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)

    # note we also have to specify the type to retrieve other wise we only get a
    # FileNode object back instead of a matrix
    K1 = cv_file.getNode("K1").mat()
    D1 = cv_file.getNode("D1").mat()
    K2 = cv_file.getNode("K2").mat()
    D2 = cv_file.getNode("D2").mat()
    R = cv_file.getNode("R").mat()
    T = cv_file.getNode("T").mat()
    E = cv_file.getNode("E").mat()
    F = cv_file.getNode("F").mat()
    R1 = cv_file.getNode("R1").mat()
    R2 = cv_file.getNode("R2").mat()
    P1 = cv_file.getNode("P1").mat()
    P2 = cv_file.getNode("P2").mat()
    Q = cv_file.getNode("Q").mat()

    cv_file.release()
    # return [K1, D1, K2, D2, R, T, E, F]
    return [K1, D1, K2, D2, R, T, E, F, R1, R2, P1, P2, Q]


def resize(src_img,scale,out_shape,offset=[],direct="None"):
    h, w, _ = src_img.shape
    s_x, e_x, s_y, e_y = int((scale[0] - 1) * w / (2 * scale[0])), int(
        (scale[0] + 1) * w / (2 * scale[0])), int(
        (scale[1] - 1) * h / (2 * scale[1])), int((scale[1] + 1) * h / (2 * scale[1]))

    if offset !=[]:
        s_y, e_y = s_y+offset[0], e_y+offset[1]

    if direct=="left":
        cropped_img = src_img[s_y:e_y, 0:e_x-s_x, :]
    elif direct=="right":
        cropped_img = src_img[s_y:e_y, w-(e_x-s_x):w, :]
    else:
        cropped_img = src_img[s_y:e_y, s_x:e_x, :]

    cropped_img = cv2.resize(cropped_img, dsize=[out_shape[1],out_shape[0]])
    return cropped_img



def depth_map(imgL, imgR):

    window_size = 5  # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
    # print(imgR.shape)
    left_matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities= 1*16,  # max_disp has to be dividable by 16 f. E. HH 192, 256
        blockSize=window_size,
        P1=8 * 3 * window_size ,
        # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
        P2=32 * 3 * window_size,
        disp12MaxDiff=7,
        uniquenessRatio=10,
        speckleWindowSize=50,
        speckleRange=32,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_HH4
    )
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    # FILTER Parameters
    lmbda = 80000
    sigma = 1.3
    visual_multiplier = 6

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)

    wls_filter.setSigmaColor(sigma)
    displ = left_matcher.compute(imgL, imgR)  # .astype(np.float32)/16
    dispr = right_matcher.compute(imgR, imgL)  # .astype(np.float32)/16
    displ = np.int16(displ)
    dispr = np.int16(dispr)
    filteredImg = wls_filter.filter(displ, imgL, None, dispr)  # important to put "imgL" here!!!

    filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
    filteredImg = np.uint8(filteredImg)

    return filteredImg,displ,dispr


if __name__ == '__main__':


    stereo_calibration_file = "calib_result/p30pro.yml"
    stereo_calibration_file = "calib_result/mate40pro_horizonal_0-0.yml"
    stereo_calibration_file = "calib_result/xm12su_1.51635581_1.51225972_221208.yml"
    K1, D1, K2, D2, R, T, E, F, R1, R2, P1, P2, Q = load_stereo_coefficients(stereo_calibration_file)
    video_dir= "images/p30_20220720"
    video_dir = "images/221205-2"
    video_dir = "images/221208-3/"
    videos = glob.glob(f"{video_dir}/*.mp4")
    print(videos)

    crop_right = [1.03855444, 1.05697224]
    crop_right = []
    crop_right = [1.51635581, 1.51225972]
    crop_left = []
    for video in videos:
        frames = []
        cap = cv2.VideoCapture(video)
        video_name = video.split(os.sep)[-1].replace(".mp4","")
        os.makedirs(f"{video_dir}/madnet_train_videos/{video_name}/",exist_ok=True)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        for i,frame in enumerate(frames[38:]):
            w_h,w_w,_ = frame.shape
            left_image,right_image = frame[:int(w_h/2),:,:],frame[int(w_h/2):,:,:]

            if "p30" not in video_dir:
                left_image = np.rot90(left_image, -1)
                right_image = np.rot90(right_image, -1)
            height, width, channel = left_image.shape  # We will use the shape for remap

            if crop_right != []:
                right_image = resize(src_img=right_image, scale=crop_right, out_shape=right_image.shape[:2],direct="right")

            if crop_left != []:
                left_image = resize(src_img=left_image, scale=crop_left, out_shape=left_image.shape[:2],direct="left")
            height, width, channel = left_image.shape  # We will use the shape for remap

            leftMapX, leftMapY = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (width, height), cv2.CV_32FC1)
            left_rectified = cv2.remap(left_image, leftMapX, leftMapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
            rightMapX, rightMapY = cv2.initUndistortRectifyMap(K2, D2, R2, P2, (width, height), cv2.CV_32FC1)
            right_rectified = cv2.remap(right_image, rightMapX, rightMapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)

            cv2.imwrite(f"{video_dir}/madnet_train_videos/{video_name}/left_{i}.png",left_rectified)
            cv2.imwrite(f"{video_dir}/madnet_train_videos/{video_name}/right_{i}.png", right_rectified)
