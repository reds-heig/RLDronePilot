import matplotlib.pyplot as plt
import numpy as np
import random as rdm
from shapely.geometry import LineString

class Line:
    def __init__(self, seed, num_points=500, max_angle_change=np.pi/9, straight_points=10, straight_points2=50, window_size=75):
        rdm.seed(seed)
        
        self.num_points = num_points
        self.max_angle_change = max_angle_change
        self.straight_points = straight_points
        self.straight_points2 = straight_points2
        self.window_size = window_size
        
        self.reset()


    def reset(self):
        path = self.generate_path(
            num_points=self.num_points, 
            max_angle_change=self.max_angle_change, 
            straight_points=self.straight_points
        )
        self.path = self.smooth_path(
            path, 
            window_size=self.window_size
        )
        # path is (z, x) with values in centimeters
        self.path_length = self.get_total_length() # in centimeters

    
    def generate_path(self, num_points=100, max_step_size=0.1, max_angle_change=np.pi/6, straight_points=10):
        # initialize starting point
        path = [(0, 0)]
    
        # generate straight path for the specified number of points
        for i in range(1, straight_points + 1):
            path.append((i * max_step_size, 0))
    
        # continue with smooth path
        for _ in range(straight_points + 1, num_points):
            # generate random step size and angle change
            step_size = np.random.uniform(0, max_step_size)
            angle_change = np.random.uniform(-max_angle_change, max_angle_change)
    
            # calculate new point based on previous point, step size, and angle change
            x, y = path[-1]
            new_x = x + step_size * np.cos(angle_change)
            new_y = y + step_size * np.sin(angle_change)
    
            path.append((new_x, new_y))
    
        return np.array(path)

    
    def smooth_path(self, path, window_size=50):
        # separate x and y coordinates
        x, y = path[:, 0], path[:, 1]
    
        # apply a simple moving average to smooth the path
        smoothed_x = np.convolve(x, np.ones(window_size)/window_size, mode='valid')
        smoothed_y = np.convolve(y, np.ones(window_size)/window_size, mode='valid')
    
        # pad the smoothed arrays to match the length of the original path
        pad_size = (len(path) - len(smoothed_x)) // 2
        smoothed_x = np.pad(smoothed_x, (pad_size, pad_size), mode='edge')
        smoothed_y = np.pad(smoothed_y, (pad_size, pad_size), mode='edge')
    
        smoothed_path = np.column_stack((smoothed_x, smoothed_y))

        # shift origin to 0,0
        smoothed_path[:, 0] = smoothed_path[:, 0] - smoothed_path[0][0]
        smoothed_path[:, 1] = smoothed_path[:, 1] - smoothed_path[0][1]

        # interpolate y-values for evenly spaced x-values
        target_step = smoothed_path[-1][0] / self.num_points
        new_x_values = np.arange(0, smoothed_path[-1, 0] + target_step, target_step)
        new_y_values = np.interp(new_x_values, smoothed_path[:, 0], smoothed_path[:, 1])

        # scale to get values in cm
        new_x_values = np.arange(len(new_x_values))
        new_y_values *= 1000

        new_x_values = np.arange(self.straight_points2 + len(new_x_values))
        new_y_values = np.concatenate((np.zeros(self.straight_points2), new_y_values))
    
        interpolated_path = np.column_stack((new_x_values, new_y_values))
    
        return interpolated_path
      
    
    def find_closest_point(self, point, return_closest_index=False):
        # point has coordinates in centimeters
        
        # calculate the Euclidean distance between the given point and all points in the path
        distances = np.linalg.norm(self.path - point, axis=1)
        
        # find the index of the closest point
        closest_index = np.argmin(distances)
        
        # get the closest point and its distance
        closest_point = self.path[closest_index]
        distance = distances[closest_index]

        if not return_closest_index:
            return closest_point, distance # closest_point coordinates & distance in centimeters
        else:
            return closest_point, distance, closest_index # closest_point coordinates & distance in centimeters


    def get_travelled_distance(self, point):
        # point has coordinates in centimeters
        
        # find closest point in the path
        closest_index = self.find_closest_point(point, return_closest_index=True)[-1]
    
        # path up to the closest point
        partial_path = self.path[:closest_index + 1]
    
        # calculate the distance along the partial path from the origin to the closest point
        distance_along_path = np.sum(np.linalg.norm(np.diff(partial_path, axis=0), axis=1))
        return distance_along_path # distance in centimeters


    def get_total_length(self):
        distance_along_path = np.sum(np.linalg.norm(np.diff(self.path, axis=0), axis=1))
        return distance_along_path # distance in centimeters


    def find_intersection(self, point1, point2):
        vector_line = LineString([point1, point2])

        for i in range(len(self.path) - 1):
            segment = LineString([self.path[i], self.path[i + 1]])
    
            # check for intersection
            if vector_line.intersects(segment):
                intersection_point = vector_line.intersection(segment)
                return intersection_point.x, intersection_point.y
    
        # if no intersection is found, return None
        return [None, None]
    
    
    def get_render_dict(self):
        return {'line-path': self.path}


    @staticmethod
    def render(params, show=True):
        plt.plot(params['line-path'][:, 0], params['line-path'][:, 1], linewidth=1)
        plt.title('Random Smooth Path')
        plt.xlabel('Z-axis (cm)')
        plt.ylabel('X-axis (cm)')
        plt.xticks(rotation='vertical')
        plt.grid(True)
        if show:
            plt.show()