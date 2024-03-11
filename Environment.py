
from collections import namedtuple
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
import numpy as np
import random as rdm
import time
from torch import Tensor

from Drone import Drone
from Line import Line


"""

Line coords are in centimeters
Drone position coords are in meters
Plots are done with values in centimeters

Actions:
- z-speed (forward), range [ 0,1] (0 = slow forward speed, 1 = fast forward speed)
- x-speed (lateral), range [-1,1] (-1 = move left fast, 0 = no lateral movement, 1 = move right fast)
- a-speed (angular), range [-1,1] (-1 = turn left fast, 0 = no turn, 1 = turn right fast)

State:
- p1.x, range [-1,1], if not found = -2
- p2.z, range [ 0,1], if not found = -2
- p2.x, range [-1,1], if not found = -2

p1: intersection between the line and the bottom border of the scene
p2: intersection between the line and either the top, left or right border of the scene

"""


class Environment():
    def __init__(self, seed, max_allowed_dist, z_min_speed, z_max_speed, x_max_speed, max_angular_speed, max_drift,
                 allow_x_movement, speed_z_activation_dist, target_dist_p1_C, alpha, beta, gamma):
        rdm.seed(seed)
        
        self.drone = Drone(seed, z_min_speed, z_max_speed, x_max_speed, max_angular_speed, max_drift, allow_x_movement)
        self.line = Line(seed, num_points=1_500)
        self.i_episode = 0

        self.episode_length = 500
        self.max_allowed_dist = max_allowed_dist # meters
        self.speed_z_activation_dist = speed_z_activation_dist
        self.target_dist_p1_C = target_dist_p1_C
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    
    def reset(self): 
        self.i_episode = 0
        self.drone.reset()
        self.line.reset()

        state = self.drone.find_line(self.line)
        assert Drone.sees_line(state), 'Line not seen from the initial position.'

        return state

    
    def step(self, action):
        # compute next state
        observation = self.drone.update(action, self.line)
        # compute reward
        reward = self.get_reward_(action[0])
        is_drone_too_far = self.line.find_closest_point([self.drone.pos[0] * 100, self.drone.pos[1] * 100])[1] / 100 > self.max_allowed_dist
        terminated = is_drone_too_far or not Drone.sees_line(observation)
        truncated = self.i_episode >= self.episode_length
        # keep track of the iteration we are in current episode
        self.i_episode += 1
        return observation, reward, terminated, truncated

    
    def sample(self):
        # randomly sample one action (return the index of this action)
        return self.drone.sample_rdm_action()


    def get_reward_(self, speed_z):
        # speed_z in range [0., 1.]
        travelled_distance = self.line.get_travelled_distance([self.drone.pos[0] * 100, self.drone.pos[1] * 100]) / 100 # travelled_distance in meters
        # normalized distance between p1 and the center of the image
        dist_p1_C = abs(self.drone.p1_normalized_x) if self.drone.p1_normalized_x is not None else None
        
        # speed_z is proportional to the travelled distance: the longer the drone flies, the more speed_z should be taken into account in the reward
        # the reward only accounts for speed_z after the drone flew a minimum of "speed_z_activation_dist" meters
        reward_A = travelled_distance * (self.alpha if travelled_distance < self.speed_z_activation_dist else 1.)
        reward_B = max(0, travelled_distance - self.speed_z_activation_dist) * self.beta * speed_z
        reward_C = (reward_A + reward_B) * np.interp(dist_p1_C, [0., 1.], [0.1, 1.]) * self.gamma
        return reward_A + reward_B + reward_C

    
    def render(self, out):
        with out:
            # render the path
            self.line.plot(show=False)
            # render the drone
            self.drone.plot(show=False)
            # render the closest point in the path from the drone
            closest_point_in_cm = self.line.find_closest_point([self.drone.pos[0] * 100, self.drone.pos[1] * 100])[0]
            plt.scatter(closest_point_in_cm[0], closest_point_in_cm[1], color='darkgreen', marker='o', s=10, label='Closest point in path')
            
            # customize the axis step
            step = 50
            ax = plt.gca()
            ax.xaxis.set_major_locator(plt.MultipleLocator(step))
            ax.yaxis.set_major_locator(plt.MultipleLocator(step))
            ax.set_aspect('equal')
            
            plt.legend(bbox_to_anchor=(1.0,1.0))
            clear_output(wait=True)
            plt.show(block=True)
            display(plt.gcf())
            plt.clf()