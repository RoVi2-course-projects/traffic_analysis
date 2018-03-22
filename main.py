#!/usr/bin/python3
# Standard libraries

# Download video
#https://www.youtube.com/watch?v=0WlKMAga7BY&feature=youtu.be

import cv2
import matplotlib.pyplot as plt
import numpy as np
#   import perspective_correction as pc
import optical_flow_tracking
from bg_extraction import BackgroundSubtractor
from optical_flow_tracking import OpticalFlowTracking
import math

def main():
    # Paths to files
    vid_path = "./resources/trafic-video.mp4"
    map_path = "./resources/googlemap.png"
    video = cv2.VideoCapture(vid_path)
    # Instantiate the main class for subtracting the background.
    bg_subtractor = BackgroundSubtractor()
    oft = OpticalFlowTracking()
    ret, frame = video.read()

    bg_subtractor.update_frame(frame)
    fgmask = bg_subtractor.extract_background()
    centroids = bg_subtractor.find_centroids()
    centroids = np.asarray(centroids)

    old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(frame)
    cnt = 0
    init=True
    while(1):
        ret, frame = video.read()
        if not ret:
            break
        # Update the video frame and refresh the background image.
        cnt = cnt + 1
        if cnt < 8:
            bg_subtractor.update_frame(frame)
            fgmask = bg_subtractor.extract_background()
            centroids = bg_subtractor.find_centroids()
            centroids = np.asarray(centroids)

        if centroids.size and cnt > 5:

            if init:
                #init_kalman()
                sz = (int(np.divide(centroids.size,2))) # size of array
                xhat = np.zeros((sz, 2))      # a posteri estimate of x
                import pdb; pdb.set_trace()
                phat = np.zeros((sz))        # a posteri error estimate
                init=False

            centroids = centroids.reshape((-1,1,2)).astype(np.float32)
            # xhat_new = xhat.reshape(-1,1,2)
            # phat_new = phat.reshape(-1,1)
            centroids, old_gray, mask, xhat, phat= oft.optical_flow_tracking(
                centroids, frame, old_gray, mask, xhat, phat)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    video.release()
    cv2.destroyAllWindows()
    return


if __name__ == "__main__":
    main()
