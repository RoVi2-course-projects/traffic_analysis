#!/usr/bin/python3
# Standard libraries
import cv2
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output

def image_transformation(frame, shape, src_points, dst_points):
    homography = find_homography(src_points,dst_points)
    transformed_img = perspective_correction(frame, shape, homography)
    
    return transformed_img

def surf_detection(frame):
    surf =  cv2.SIFT_create()
    key_points, descriptors = surf.detectAndCompute(frame, None)
    frame = cv2.drawKeypoints(frame, key_points, None, (255,0,0), 4)
    
    return frame

def playvideowin(vid, shape, src_points, dst_points, winname='video'):
    while(1):
        ret, frame = vid.read()
        if not ret:
            vid.release()
            print("Released Video Resource")
            break
    
        frame = image_transformation(frame,shape,src_points,dst_points)
        frame = surf_detection(frame)

        cv2.namedWindow(winname)
        cv2.startWindowThread() #this normally isn't required
        cv2.imshow(winname,frame)
        k=cv2.waitKey(10)
        
        if k==27: #exit is Esc is pressed
            break

    cv2.destroyAllWindows() 

def perspective_correction(frame, shape, homography):
    transformed_frame = cv2.warpPerspective(frame, homography, 
                                            (shape[1], shape[0]))

    return transformed_frame
    
def find_homography(src_points,dst_points):
    homography, status = cv2.findHomography(src_points,dst_points)    

    return homography
    
def get_map_shape(path):
    map_img = cv2.imread(path, -1) 
    shape = map_img.shape
    
    return shape
    