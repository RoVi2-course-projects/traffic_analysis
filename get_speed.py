#!/usr/bin/python3
import math
import numpy as np

# obtain old and new pixels to track
# ground sampling distance is in cm/pixels
def calculate_speed(old_pos_x,old_pos_y, new_pos_x, new_pos_y,
                    ground_sampling_distance, fps):
    # scale converts GDS to meters/pixels
    scale = np.divide(ground_sampling_distance, 100.0)

    dist_between_coordinates = math.sqrt(pow(new_pos_x-old_pos_x,2) +
                                         pow(new_pos_y-old_pos_y,2))

    # pixel/frame * frame/second * meter/pixel - returns meters/second
    speed = round(dist_between_coordinates * fps * scale,1)
    # convert to km/t
    speed = speed * 3.6
    return speed


if __name__ == "__main__":

    #test
    fps = 25.0 # estimated from video data
    gds = 50.0 # estimated from video still picture
    speed = calculate_speed(1,2,3,4,gds,fps)
    print("get speed", speed)
