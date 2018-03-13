#!/usr/bin/python3
# Standard libraries
import cv2
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output


def playvideowin(vidname,winname='video'):
    vid = cv2.VideoCapture(vidname)
    while(1):
        ret, frame = vid.read()
        if not ret:
            vid.release()
            print("Released Video Resource")
            break
        
        #homography = find_homography(frame)
        #frame = perspective_correction(frame, homography)

        cv2.namedWindow(winname)
        cv2.startWindowThread() #this normally isn't required
        cv2.imshow(winname,frame)
        k=cv2.waitKey(100)
        
        if k==27: #exit is Esc is pressed
            break

    cv2.destroyAllWindows()   


def perspective_correction(frame, homography):
    transformed_frame = cv2.warpPerspective(frame, homography, 
                                            (frame.shape[1], frame.shape[0]))

    return transformed_frame
    
def find_homography(frame, angle = 0):
    height, width, channels = frame.shape
    offset = 200
    dst_points = np.array([[0,0],[width,0],[0+offset,height],
                           [width-offset,height]])
                           
    src_points = np.array([[0,0],[width,0],[0,height],[width,height]])
    homography, status = cv2.findHomography(src_points,dst_points)    

    return homography

if __name__ == "__main__":
    
    #playvideowin("../videos/2017_06_22_1439 Krydset Falen Kløvermosevej.mp4")
    vid = cv2.VideoCapture("../videos/2017_06_22_1439 Krydset Falen Kløvermosevej.mp4")
    ret, frame = vid.read()    
    plt.figure("tokyo")
    plt.imshow(frame)
    plt.show()