
from collections import namedtuple
import math
import matplotlib.pyplot as plt
import numpy as np
import random as rdm

class Drone:

    # DO NOT CHANGE FOLLOWING CONSTANTS
    _height = 0.2 # meters
    
    _fov = math.radians(87) # radians
    _focal_length = 0.35 # meters

    _p1_dist = _height / np.tan(_fov/2) # distance in meters: drone - bottom of image
    _img_width_p1 = np.tan(_fov/2) * _p1_dist * 2 # meters
    _p2_dist = 0.70 # meters
    _img_width_p2 = np.tan(_fov/2) * _p2_dist * 2 # meters

    _fps = 5 # approximation
    _flight_commands_rate = 0.1 # seconds;  a new command is given to the drone every 0.1 second
    _n_flight_commands = int((1/_fps) / _flight_commands_rate) # number of times a flight command is repeated before the next one comes up

    _line_not_seen_placeholder = -2
    # ---

    
    def __init__(self, seed, z_min_speed, z_max_speed, x_max_speed, max_angular_speed, max_drift, allow_x_movement):
        rdm.seed(seed)
        self.z_min_speed = z_min_speed             # meters / second
        self.z_max_speed = z_max_speed             # meters / second
        self.x_max_speed = x_max_speed             # meters / second
        self.max_angular_speed = max_angular_speed # degrees / seconds
        self.max_drift = max_drift                 # meters
        self.allow_x_movement = allow_x_movement
        self.reset()


    def reset(self):
        self.pos = [0., 0.] # (z, x) in meters
        self.dir = 0. # degrees
    

    def update(self, action, line):
        if self.allow_x_movement:
            prev_speed_z, prev_speed_x, prev_speed_a = action
        else:
            prev_speed_z, prev_speed_a = action
            prev_speed_x = 0.

        # change the range of the speed values
        mapped_prev_speed_z, mapped_prev_speed_x, mapped_prev_speed_a = self.pilot_speeds_to_drones(prev_speed_z, prev_speed_x, prev_speed_a)

        # update position & direction
        for _ in range(self._n_flight_commands):
            angle_radians = math.radians(self.dir)

            delta_z = mapped_prev_speed_z * math.cos(angle_radians) + mapped_prev_speed_x * math.sin(angle_radians)
            delta_x = mapped_prev_speed_z * math.sin(angle_radians) - mapped_prev_speed_x * math.cos(angle_radians)
            self.pos[0] = self.pos[0] + delta_z # update z pos
            self.pos[1] = self.pos[1] + delta_x # update x pos

            # apply a random drift to the drone
            rdm_drift = (
                rdm.uniform(-1,1) * self.max_drift,
                rdm.uniform(-1,1) * self.max_drift,
            )
            self.pos[0] = self.pos[0] + rdm_drift[0]
            self.pos[1] = self.pos[1] + rdm_drift[1]
            
            self.dir += mapped_prev_speed_a # update direction (angle in degrees)

        p1_pixel_x, p2_pixel_z, p2_pixel_x = self.find_line(line) # get coordinates in pixels (in the image) of the points from the line seen by the drone

        observation = [
            p1_pixel_x,             # curr_line_p1 (closer point)
            p2_pixel_z, p2_pixel_x, # curr_line_p2 (further point)
        ]
        return observation


    def pilot_speeds_to_drones(self, speed_z, speed_x, speed_a):
        # speed_z [ 0., 1.] -> [z_min_speed, z_max_speed]
        # speed_x [-1., 1.] -> [-x_max_speed, x_max_speed]
        # speed_a [-1., 1.] -> [max_angular_speed, -max_angular_speed]
        mapped_speed_z = np.interp(speed_z, [ 0., 1.], [ self.z_min_speed, self.z_max_speed])
        mapped_speed_x = np.interp(speed_x, [-1., 1.], [-self.x_max_speed, self.x_max_speed])
        mapped_speed_a = np.interp(speed_a, [-1., 1.], [self.max_angular_speed, -self.max_angular_speed])
        return mapped_speed_z, mapped_speed_x, mapped_speed_a


    def find_line(self, line):
        """
        Returns p1.x, p2.z, and p2.x.
        p1: intersection point between the line and the bottom border of the scene
        p2: intersection point between the line and either the top, left or right border of the scene
        All coordinates are normalized.
        - p1.x, range [-1,1], if not found = -2
        - p2.z, range [ 0,1], if not found = -2
        - p2.x, range [-1,1], if not found = -2
        """
        # find direction vector (unit)
        angle_radians = math.radians(self.dir)
        self.unit_vector_dir = (
            math.cos(angle_radians),
            math.sin(angle_radians),
        )

        # find p1
        self.bottom_left_img_point, self.bottom_right_img_point = self.__get_img_point_and_segment(self.unit_vector_dir, self._p1_dist, self._img_width_p1) # in centimeters
        self.p1 = line.find_intersection(self.bottom_left_img_point, self.bottom_right_img_point) # in centimeters

        # find p2
        self.top_left_img_point, self.top_right_img_point = self.__get_img_point_and_segment(self.unit_vector_dir, self._p2_dist, self._img_width_p2) # in centimeters
        self.p2 = line.find_intersection(self.top_left_img_point, self.top_right_img_point) # in centimeters

        p2_side = None # None if p2 is on the top border of the scene; 1 if p2 is on the right border of the scene; 0 if p2 is on the left border of the scene
        
        if None in self.p2:
            # no intersection found with upper border

            # search for intersections with left and right image borders
            p2_l = line.find_intersection(self.top_left_img_point, self.bottom_left_img_point) # in centimeters
            p2_r = line.find_intersection(self.top_right_img_point, self.bottom_right_img_point) # in centimeters

            if None in p2_l and not None in p2_r:
                # no intersection found with the left border
                # but one intersection found with the right border
                self.p2 = p2_r
                p2_side = 1
            elif not None in p2_l and None in p2_r:
                # no intersection found with the right border
                # but one intersection found with the left border
                self.p2 = p2_l
                p2_side = 0
            elif not None in p2_l and not None in p2_r:
                # intersections found with both borders
                # keep the intersection further away from the drone (z coordinate)
                if self.__dist_points(self.bottom_left_img_point, p2_l) > self.__dist_points(self.bottom_right_img_point, p2_r):
                    self.p2 = p2_l
                    p2_side = 0
                else:
                    self.p2 = p2_r
                    p2_side = 1
            else:
                # no intersection found with either left or right borders
                pass

        # normalize p1 and p2
        # also we discard the z coordinate of p1 to reduce the complexity
        self.p1_normalized_x, p2_normalized_z, p2_normalized_x = \
            self._line_not_seen_placeholder, self._line_not_seen_placeholder, self._line_not_seen_placeholder
        
        Q = self.bottom_left_img_point
        R = self.top_left_img_point
        S = self.top_right_img_point
        T = self.bottom_right_img_point

        if not None in self.p1:
            self.p1_normalized_x = (self.__dist_points(Q, self.p1) / self.__dist_points(Q, T)) * 2. - 1 # in range [-1, 1], -1=left, 1=right

        if not None in self.p2:
            if p2_side == None: # p2 is on the top border of the scene
                p2_normalized_z = 1.
                p2_normalized_x = (self.__dist_points(R, self.p2) / self.__dist_points(R, S)) * 2. - 1 # in range [-1, 1], -1=left, 1=right
            else:
                if p2_side == 1: # p2 is on the right border of the scene
                    p2_normalized_x = 1. # in range [-1, 1], -1=left, 1=right
                    p2_prime = T
                    p2_second = S
                else: # p2 is on the left border of the scene
                    p2_normalized_x = -1. # in range [-1, 1], -1=left, 1=right
                    p2_prime = Q
                    p2_second = R
                p2_normalized_z = self.__dist_points(p2_prime, self.p2) / self.__dist_points(p2_prime, p2_second) # in range [0, 1], 0=close to bottom of scene, 1=far from bottom of scene

        return self.p1_normalized_x, p2_normalized_z, p2_normalized_x


    def get_render_dict(self):
        # drone position
        pos_in_cm = (
            self.pos[0] * 100,
            self.pos[1] * 100,
        )
        return {
            'drone-pos_in_cm': pos_in_cm,
            'drone-unit_vector_dir': self.unit_vector_dir,
            'drone-bottom_left_img_point': self.bottom_left_img_point,
            'drone-bottom_right_img_point': self.bottom_right_img_point,
            'drone-top_left_img_point': self.top_left_img_point,
            'drone-top_right_img_point': self.top_right_img_point,
            'drone-p1': self.p1,
            'drone-p2': self.p2,
        }


    def __get_img_point_and_segment(self, unit_vector_dir, dist, img_width):
        # find the point in the bottom center of the image
        center_img_point = (
            (self.pos[0] + unit_vector_dir[0] * dist) * 100, # *100 to get coords in cm
            (self.pos[1] + unit_vector_dir[1] * dist) * 100, # *100 to get coords in cm
        )

        # perpendicular vector which is the "line" representing the bottom of the image
        img_vector = (unit_vector_dir[1], -unit_vector_dir[0])
        img_vector = (
            img_vector[0] * img_width/2 * 100, # *100 to get coords in cm
            img_vector[1] * img_width/2 * 100, # *100 to get coords in cm
        )

        # find left and right points of the segment
        left_img_point = (
            center_img_point[0] - img_vector[0],
            center_img_point[1] - img_vector[1],
        )
        right_img_point = (
            center_img_point[0] + img_vector[0],
            center_img_point[1] + img_vector[1],
        )
        return left_img_point, right_img_point


    def __dist_points(self, point1, point2):
        # euclidean distance between two points
        x1, y1 = point1
        x2, y2 = point2
        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        return distance

    
    @staticmethod
    def render(params, show=True):
        plt.scatter(params['drone-pos_in_cm'][0], params['drone-pos_in_cm'][1], color='green', marker='x', s=100, label='Drone')
        # drone direction
        if params['drone-unit_vector_dir'] is not None:
            plt.plot(
                [params['drone-pos_in_cm'][0], params['drone-pos_in_cm'][0] + params['drone-unit_vector_dir'][0] * 75],
                [params['drone-pos_in_cm'][1], params['drone-pos_in_cm'][1] + params['drone-unit_vector_dir'][1] * 75],
                color='lime', label='Direction'
            )
        # scene borders
        if params['drone-bottom_left_img_point'] is not None and params['drone-bottom_right_img_point'] is not None and \
           params['drone-top_left_img_point'] is not None and params['drone-top_right_img_point'] is not None:
            plt.plot(
                [params['drone-bottom_left_img_point'][0], params['drone-bottom_right_img_point'][0]], 
                [params['drone-bottom_left_img_point'][1], params['drone-bottom_right_img_point'][1]], 
                color='red', alpha=0.5, label='Scene borders'
            )
            plt.plot(
                [params['drone-top_left_img_point'][0], params['drone-top_right_img_point'][0]], 
                [params['drone-top_left_img_point'][1], params['drone-top_right_img_point'][1]], 
                color='red', alpha=0.5,
            )
            plt.plot(
                [params['drone-bottom_left_img_point'][0], params['drone-top_left_img_point'][0]], 
                [params['drone-bottom_left_img_point'][1], params['drone-top_left_img_point'][1]], 
                color='red', alpha=0.5,
            )
            plt.plot(
                [params['drone-bottom_right_img_point'][0], params['drone-top_right_img_point'][0]], 
                [params['drone-bottom_right_img_point'][1], params['drone-top_right_img_point'][1]], 
                color='red', alpha=0.5,
            )
        # line (p1 and p2 are already in centimeters)
        if params['drone-p1'] is not None and not None in params['drone-p1']:
            plt.scatter(params['drone-p1'][0], params['drone-p1'][1], color='darkblue', marker='o', s=30, label='Line - p1')
        if params['drone-p2'] is not None and not None in params['drone-p2']:
            plt.scatter(params['drone-p2'][0], params['drone-p2'][1], color='darkblue', marker='o', s=30, label='Line - p2')
        # line
        if show:
            plt.show()
    
    
    @staticmethod
    def sees_line(state):
        # returns True only if both points of the line are seen by the drone
        return not Drone._line_not_seen_placeholder in state