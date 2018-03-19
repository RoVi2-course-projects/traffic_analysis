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


def main():
    # Paths to files
    vid_path = "./resources/trafic-video.mp4"
    map_path = "./resources/googlemap.png"
    video = cv2.VideoCapture(vid_path)
    # Instantiate the main class for subtracting the background.
    bg_subtractor = BackgroundSubtractor()
    oft = OpticalFlowTracking()
    ret, frame = video.read()

    old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(frame)

    while(1):
        ret, frame = video.read()
        if not ret:
            break
        # Update the video frame and refresh the background image.
        bg_subtractor.update_frame(frame)
        fgmask = bg_subtractor.extract_background()
        centroids = bg_subtractor.find_centroids()
        centroids = np.asarray(centroids)
        print(type(centroids))
        print(centroids)

        centroids, old_gray, mask = oft.optical_flow_tracking(centroids, frame, old_gray, mask)

        # # Draw a circcle surrounding every detected moving shape.
        # for centroid in centroids:
        #     cv2.circle(frame, centroid, 10, (255, 0, 0),
        #                thickness=2)
        # cv2.imshow('frame', frame)
        # cv2.imshow('frame2', fgmask)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    video.release()
    cv2.destroyAllWindows()
    return

    # # Load video and shape of the map
    # vid = cv2.VideoCapture(vid_path)
    # shape = pc.get_map_shape(map_path)
    #
    # # Perspectiv transformation points, ROI
    # src_points = np.array([[74,347],[604,193],[508,60],[309,94]])
    # dst_points = np.array([[31,511],[892,451],[963,29],[444,67]])
    #
    # # Play video
    # pc.playvideowin(vid, shape, src_points, dst_points)



if __name__ == "__main__":
    main()
