#!/usr/bin/python3
import numpy as np
import cv2
import math
import get_speed
import time
from kalman_filtering_speed import kalman_filter

class OpticalFlowTracking:
    def draw_optical_flow(self, good_new, good_old, mask, frame):
        xhat = 0.0 # initial guess, estimate of x
        p = 1.0 # initial guess, error estimate
        kf = kalman_filter()
        for i,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()

            speed = get_speed.calculate_speed(a, b, c, d, 50, 25)
            xhat, p = kf.kalman_filter_speed(speed, xhat, p)

            mask = cv2.line(mask, (a,b),(c,d),(0,0,255), 2)
            #frame = cv2.circle(frame,(a,b),9,color[i].tolist(),-1)
            frame = cv2.circle(frame,(a,b),9,(0,0,255),2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame,str(xhat)[0:4],(a,b), font, 1,(255,255,255),
                        2,cv2.LINE_AA)
        return mask, frame

    # inspiration found from git files referenced here:
    # https://docs.opencv.org/3.3.1/d7/d8b/tutorial_py_lucas_kanade.html
    def optical_flow_tracking(self, list_of_poi, frame, old_gray, mask0):
        # Parameters for lucas kanade optical flow
        lk_params = dict( winSize  = (15,15),
                          maxLevel = 2,
                          criteria = (cv2.TERM_CRITERIA_EPS |
                                      cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        mask = mask0
        p0 = list_of_poi
        old_g = old_gray
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_g, frame_gray, p0,
                                               None, **lk_params)
        # Select good points
        good_new = p1[st==1]
        good_old = p0[st==1]

        # draw the tracks and speed
        mask, frame = self.draw_optical_flow(good_new, good_old, mask, frame)
        img = cv2.add(frame,mask)
        cv2.imshow('frame',img)

        # Now update the previous frame and previous points
        old_g = frame_gray.copy()
        p0 = good_new.reshape(-1,1,2)

        return p0, old_g, mask

if __name__ == "__main__":


    #test
    oft = OpticalFlowTracking()
    cap = cv2.VideoCapture('./resources/trafic-video.mp4')

    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 100,
                           qualityLevel = 0.3,
                           minDistance = 7,
                           blockSize = 7 )

    # # Create some random colors
    # color = np.random.randint(0,255,(100,3))
    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

    #p0 = p0[38:40,:,:]
    #p0 = p0[80:82,:,:] # for other cars
    poi = p0
    mask = np.zeros_like(old_frame)
    while(1):

        ret,frame = cap.read()
        poi, old_gray, mask = oft.optical_flow_tracking(poi, frame,
                                                    old_gray, mask)

        print(type(poi))
        print(poi)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    cv2.destroyAllWindows()
    cap.release()
