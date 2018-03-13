# obtain old and new pixels to track
# ground sampling distance is in cm/pixels
def calculate_speed(old_pos_x,old_pos_y, new_pos_x, new_pos_y,
                    grund_sampling_distance):
    dist_between_coordinates = math.sqrt(pow(new_pos_x-old_pos_x,2) +
                                         pow(new_pos_y-old_pos_y,2))
    speed = round(dist_between_coordinates * fps * scale,1)
    return speed
