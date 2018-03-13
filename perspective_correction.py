#!/usr/bin/python3
# Standard libraries
import cv2
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output

def playvideowin(vidname, shape, src_points, dst_points, winname='video'):
    vid = cv2.VideoCapture(vidname)
    while(1):
        ret, frame = vid.read()
        if not ret:
            vid.release()
            print("Released Video Resource")
            break
    
        homography = find_homography(src_points,dst_points)
        frame = perspective_correction(frame, shape, homography)

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

if __name__ == "__main__":
    
    map_reference_BGR = cv2.imread("./images/googlemap.png") 
    shape = map_reference_BGR.shape
    src_points = np.array([[74,347],[604,193],[508,60],[309,94]])   
    dst_points = np.array([[31, 511],[892,451],[963,29],[444,67]])
    playvideowin("../videos/videoplayback", shape)
    
    """
    plt.figure("tokyo")
    plt.imshow(frame)
    plt.show()
    """