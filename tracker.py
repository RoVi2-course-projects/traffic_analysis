#!/usr/bin/python3
"""
Module for implementing tracking algorithms.

TODO: Create a class containing the background subtractor and its
parameters.
Create as well a method for detecting contours and their centroids.
"""
# Standard libraries
import numpy as np
# Third party libraries
import cv2


def extract_background(subtractor, frame, 
                       kernel=np.array([[0,1,0], [1,1,1],
                                       [0,1,0]]).astype(np.uint8)):
    """
    Get a binnarized image with the foreground (1s) and background (0s).
    """
    fg_mask = subtractor.apply(frame)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    return fg_mask


def main(video_path="./resources/trafic-video.mp4"):
    video = cv2.VideoCapture(video_path)
    fgbg = cv2.bgsegm.createBackgroundSubtractorMOG(history=500)
    while(1):
        ret, frame = video.read()
        if not ret:
            break
        fgmask = extract_background(fgbg, frame)
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