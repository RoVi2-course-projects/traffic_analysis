#!/usr/bin/python3
"""
Module for implementing tracking algorithms.

The main element is a class containing the background subtractor and its
parameters, as well as a method for detecting contours and their
centroids coordinates.
"""
# Standard libraries
import numpy as np
# Third party libraries
import cv2


class BackgroundSubtractor(object):
    def __init__(self, h=500, kernel=np.array([[0,1,0], [1,1,1],
                                       [0,1,0]]).astype(np.uint8)):
        # Current working frame and binarized image with the background.
        self._frame = None
        self._fg_mask = None
        # Kernel for the morphological operation after binarizing.
        self.kernel = kernel
        # Subtractor algorithm for getting the background.
        self._subtractor = cv2.bgsegm.createBackgroundSubtractorMOG(history=h)
        # List with the coordinates of all the detected centroids.
        self._centroids = []

    def update_frame(self, frame):
        """Update the actual working frame for detecting the bg."""
        self._frame = frame

    def extract_background(self):
        """Get a binnarized image where the foreground is 1."""
        self._fg_mask = self._subtractor.apply(self._frame)
        self._fg_mask = cv2.morphologyEx(self._fg_mask, cv2.MORPH_OPEN,
                                         self.kernel)
        return self._fg_mask

    def find_centroids(self):
        """
        Find the centroids (X,Y) coordinates of all detected contours.
        """
        _, contours, _ = cv2.findContours(self._fg_mask, cv2.RETR_LIST,
                                          cv2.CHAIN_APPROX_SIMPLE)
        centroids = []
        # The centroid of each contour is the arithmetic mean of all of its
        # points X and Y coordinates.
        for cnt in contours:
            centroid = cnt.mean(axis=0)[0].astype(np.uint16)
            # centroids.append(centroid)
            # Store the centroid as a tuple, for further usage with cv.circle
            centroids.append((centroid[0], centroid[1]))
        self._centroids = centroids
        return self._centroids


def main(video_path="./resources/trafic-video.mp4"):
    video = cv2.VideoCapture(video_path)
    # Instantiate the main class for subtracting the background.
    bg_subtractor = BackgroundSubtractor()
    while(1):
        ret, frame = video.read()
        if not ret:
            break
        # Update the video frame and refresh the background image.
        bg_subtractor.update_frame(frame)
        fgmask = bg_subtractor.extract_background()
        centroids = bg_subtractor.find_centroids()
        # Draw a circcle surrounding every detected moving shape.
        for centroid in centroids:
            cv2.circle(frame, centroid, 10, (255, 0, 0),
                       thickness=2)
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
