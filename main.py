#!/usr/bin/python3
"""
Program for extracting moving vehicles in a traffic lane.

Download video at:
https://www.youtube.com/watch?v=0WlKMAga7BY&feature=youtu.be
"""
# Standard libraries
import matplotlib.pyplot as plt
import numpy as np
# Third-party libaries
import cv2
# Local libraries
from bg_extraction import BackgroundSubtractor
from optical_flow_tracking import OpticalFlowTracking
import perspective_correction as pc


def main(vid_path="./resources/trafic-video.mp4",
         ref_img_path="./resources/transformed_img.jpg"): 
    # Load reference image for stabilization.
    reference_img = cv2.imread(ref_img_path, -1)
    reference_img = pc.mask_reference_img(reference_img)    
   
    # Load video
    video = cv2.VideoCapture(vid_path)
   
    # Instantiate the main class for subtracting the background.
    bg_subtractor = BackgroundSubtractor()
    oft = OpticalFlowTracking()
    ret, frame = video.read()
    
    frame = pc.surf_detection(frame, reference_img)

    old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(frame)

    while(1):
        ret, frame = video.read()
        if not ret:
            break
      
        frame = pc.surf_detection(frame, reference_img)

        # Update the video frame and refresh the background image.
        bg_subtractor.update_frame(frame)
        bg_subtractor.extract_background()
        centroids = bg_subtractor.find_centroids()
        centroids = np.asarray(centroids)

        if centroids.size:
            centroids = centroids.reshape((-1,1,2)).astype(np.float32)
            centroids, old_gray, mask = oft.optical_flow_tracking(centroids,
                    frame, old_gray, mask)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    video.release()
    cv2.destroyAllWindows()
    return


if __name__ == "__main__":
   main()
