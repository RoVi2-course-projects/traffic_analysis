#!/usr/bin/python3
"""
Module for implementing tracking algorithms.
"""
# Standard libraries
import numpy as np
# Third party libraries
import cv2


def main(video_path="./resources/trafic-video.mp4"):
    # src_points = np.array([[74,347],[604,193],[508,60],[309,94]])
    # src_points_inv = np.array([[347, 74],[193, 604],[60, 508],[94, 309]])
    video = cv2.VideoCapture(video_path)
    fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    while(1):
        ret, frame = video.read()
        # fgmask = fgbg.apply(frame)
        fgmask = fgbg.apply(frame)
        # fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        cv2.imshow('frame', frame)
        cv2.imshow('frame2', fgmask)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    video.release()
    cv2.destroyAllWindows()
    return

if __name__ == "__main__":
    main()