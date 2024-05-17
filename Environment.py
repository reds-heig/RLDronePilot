
from collections import namedtuple
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
Plots are made from values in centimeters

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


class Environment:
    def __init__(self, seed, max_allowed_dist, z_min_speed, z_max_speed, x_max_speed, max_angular_speed, max_drift,
                 allow_x_movement, alpha, beta, gamma, delta, terminated_reward):
        rdm.seed(seed)

        self.neptune = None # neptune object for logging
        drone_params = {
            'seed': seed, 
            'z_min_speed': z_min_speed, 
            'z_max_speed': z_max_speed, 
            'x_max_speed': x_max_speed, 
            'max_angular_speed': max_angular_speed, 
            'max_drift': max_drift, 
            'allow_x_movement': allow_x_movement
        }
        self.drone = Drone(**drone_params)
        line_params = {
            'seed': seed,
            'num_points': 1_500,
        }
        self.line = Line(**line_params)
        self.i_episode = 0

        self.episode_length = self.line.path_length * 50 # path length in centimeters x 50 = maximum number of iterations per episode
        self.max_allowed_dist = max_allowed_dist # meters
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.terminated_reward = terminated_reward


    def set_neptune(self, neptune):
        self.neptune = neptune

    
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
        # compute reward & check conditions
        is_drone_too_far = self.line.find_closest_point([self.drone.pos[0] * 100, self.drone.pos[1] * 100])[1] / 100 > self.max_allowed_dist
        terminated = is_drone_too_far or not Drone.sees_line(observation)
        truncated = self.i_episode >= self.episode_length
        reward = self.get_reward_(action[0], terminated)
        # keep track of the iteration we are in the current episode
        self.i_episode += 1
        return observation, reward, terminated, truncated


    def get_reward_(self, speed_z, terminated):
        
        # An optimal reward function is yet to be found.
        # Simply rewarding "alive time" remains the most
        # effective way for the agents to learn a decent
        # strategy. Feel free to experiment with this.
        return 1.
        
        # speed_z in range [ 0., 1.]
        # travelled_distance in meters
        travelled_distance = self.line.get_travelled_distance([self.drone.pos[0] * 100, self.drone.pos[1] * 100]) / 100
        # normalized distance between p1 and the center of the image; in range [0., 1.]
        dist_p1_C = abs(self.drone.p1_normalized_x) if not None in self.drone.p1 else None
        # drone_angle and line_angle; in range [-1., 1.]
        drone_angle = (np.arctan(self.drone.unit_vector_dir[0] / self.drone.unit_vector_dir[1]) / (np.pi/2)) \
                        if not None in self.drone.unit_vector_dir else None
        line_angle = (np.arctan((self.drone.p2[0] - self.drone.p1[0]) / (1e-5+(self.drone.p2[1] - self.drone.p1[1]))) / (np.pi/2)) \
                        if not None in [*self.drone.p1, *self.drone.p2] else None

        # travelled_distance_percentage in range [0., 1.]
        travelled_distance_percentage = travelled_distance / self.line.path_length
    
        reward_A = travelled_distance_percentage * self.alpha
        reward_B = travelled_distance_percentage * (speed_z * self.beta)
        reward_C = (travelled_distance_percentage * (1 - dist_p1_C) * self.gamma) \
                    if dist_p1_C is not None else 0.
        reward_D = (travelled_distance_percentage * (1 - (abs(drone_angle - line_angle) / 2)) * self.delta) \
                    if not None in [drone_angle, line_angle] else 0.

        reward = (reward_A + reward_B + reward_C + reward_D) if not terminated else self.terminated_reward
        if self.neptune is not None:
            self.neptune['train/reward_A'].log(reward_A)
            self.neptune['train/reward_B'].log(reward_B)
            self.neptune['train/reward_C'].log(reward_C)
            self.neptune['train/reward_D'].log(reward_D)
            self.neptune['train/reward'].log(reward)
        return reward


    def get_render_dict(self):
        env_render_dict = {
            'env-closest_point_in_cm': self.line.find_closest_point([self.drone.pos[0] * 100, self.drone.pos[1] * 100])[0],
        }
        return dict(**self.line.get_render_dict(), **self.drone.get_render_dict(), **env_render_dict)


    @staticmethod
    def render(params):
        # render the path
        Line.render(params, show=False)
        # render the drone
        Drone.render(params, show=False)
        # render the closest point in the path from the drone
        plt.scatter(params['env-closest_point_in_cm'][0], params['env-closest_point_in_cm'][1], 
                    color='darkgreen', marker='o', s=10, label='Closest point in path')
        
        # customize the axis step
        step = 50
        ax = plt.gca()
        ax.xaxis.set_major_locator(plt.MultipleLocator(step))
        ax.yaxis.set_major_locator(plt.MultipleLocator(step))
        ax.set_aspect('equal')
        
        plt.legend(bbox_to_anchor=(1.0,1.0))