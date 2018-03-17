#!/usr/bin/python3
import math
import numpy as np

# simple kalman filter
# inspired by http://scipy-cookbook.readthedocs.io/items/KalmanFiltering.html
class kalman_filter:

    def __init__(self):
        self.q = 1e-5 # process variance - guess
        self.r = 0.1**2 # estimate of measurement variance

    def kalman_filter_speed(self, speed, x_hat_input, p_input):
        xhat = x_hat_input
        p = p_input
        # time update

        xhat_minus = xhat
        p_minus = p + self.q

        # measurement update
        k = p_minus/( p_minus + self.r )
        xhat = xhat_minus + k * (speed -xhat_minus)
        p = (1 - k) * p_minus
        return xhat, p

if __name__ == "__main__":

    #test
    xhat = 0.0 # initial guess, estimate of x
    P = 1.0 # initial guess, error estimate
    kf = kalman_filter()
    for i in range(10):
        xhat, P = kf.kalman_filter_speed(10, xhat, P)
        print("xhat: ",xhat,"P:",P)
