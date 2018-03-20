#!/usr/bin/python3
# Standard libraries

# Download video 
#https://www.youtube.com/watch?v=0WlKMAga7BY&feature=youtu.be

import cv2
import matplotlib.pyplot as plt
import numpy as np
import perspective_correction as pc
  

def main():
    # Paths to files
    vid_path = "../videos/videoplayback"
    map_path = "./resources/googlemap.png"
    ref_img_path = "./resources/transformed_img.jpg"   
 
    # Load video and shape of the map
    vid = cv2.VideoCapture(vid_path)
    reference_img = cv2.imread(ref_img_path, -1)

    reference_img = pc.mask_reference_img(reference_img)    

    #cv2.imshow('image',reference_img)
    #cv2.waitKey(0)

    # Perspectiv transformation points, ROI
    src_points = np.array([[74,347],[604,193],[508,60],[309,94]])   
    dst_points = np.array([[31,511],[892,451],[963,29],[444,67]])
    
    # Play video
    pc.playvideowin(vid, reference_img, src_points, dst_points)

if __name__ == "__main__":
   main()
