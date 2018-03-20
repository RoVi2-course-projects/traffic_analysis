#!/usr/bin/python3
# Standard libraries
import cv2
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output

def mask_reference_img(img):
    points_set1 = np.array([[400,0],[400,535],[650,535],[600,0]])
    points_set2 = np.array([[0,520],[0,535],[340,535],[999,350],[999,170]])
    cv2.fillConvexPoly(img, np.int32([points_set1]), (0,0,0))
    cv2.fillConvexPoly(img, np.int32([points_set2]), (0,0,0))
    
    return img

def image_transformation(frame, shape, src_points, dst_points):
    homography, mask = cv2.findHomography(src_points,dst_points,cv2.RANSAC,10.0)
    transformed_img = perspective_correction(frame, (shape[0],shape[1]), homography)
    
    return transformed_img

def GSD_px_to_meters(pixels):
    p1 = [437,373]
    p2 = [604,312]
    googlemap_distance = 38.39 #[meters]
    r = sqrt(pow(p1[0]-p2[0],2) + pow(p1[1]-p2[1],2))
    scale = r/googlemap_distance

    meters = pixels/scale
    return meters

def surf_detection(frame, reference_image):
    surf =  cv2.xfeatures2d.SIFT_create()
    key_points1, descriptors1 = surf.detectAndCompute(frame, None)
    key_points2, descriptors2 = surf.detectAndCompute(reference_image, None) 

    bf = cv2.BFMatcher()
    #matches = bf.match(descriptors1,descriptors2)
    matches = bf.knnMatch(descriptors1,descriptors2,k=2)

    good = []
    for m,n in matches:    
        if m.distance < 0.8*n.distance:
            good.append(m)
   
    src_points = np.float32([ key_points1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_points = np.float32([ key_points2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    #cv2.drawKeypoints(frame, good, frame, (0,0,255),4) 
    #for point in src_points:
    #    cv2.circle(frame, (point[0][0],point[0][1]), 5, (0,0,255)) 
    return image_transformation(frame, reference_image.shape, src_points, dst_points)


def playvideowin(vid, reference_image, src_points, dst_points, winname='video'):
    
    while(1):
        ret, frame = vid.read()
        if not ret:
            vid.release()
            print("Released Video Resource")
            break
        
        frame = surf_detection(frame, reference_image) 

        cv2.namedWindow(winname)
        cv2.startWindowThread() #this normally isn't required
        cv2.imshow(winname,frame)
        k=cv2.waitKey(2)
        
        if k==27: #exit is Esc is pressed
            break

    #cv2.imwrite("./resources/transformed_img.jpg", frame);
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
    
