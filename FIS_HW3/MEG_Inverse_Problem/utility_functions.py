import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math


mu = 4 * math.pi * pow(10, -7)

#  -------------------------------------  Sort points based on distance from z-axis  -------------------------------------

def sorted(x, y, z):
    # Calculate distance from z-axis for each point
    distances = np.sqrt(x**2 + y**2)
    sorted_indices = np.argsort(distances)
    x_sorted = x[sorted_indices]
    y_sorted = y[sorted_indices]
    z_sorted = z[sorted_indices]
    return x_sorted, y_sorted, z_sorted

#  ------------------------------- Convert spherical coordinates to Cartesian coordinates -------------------------------------
def Conv_coordinates(phi, theta, radius):

    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)
    return x, y, z

def Calc_G(er, r, rq):

    a_ij = (mu / (4 * np.pi)) * (np.cross(er, rq)) / (np.linalg.norm(r - rq))

    return a_ij

#  ------------------------------- Convert Cartesian coordinates to spherical coordinates -------------------------------------

def cartesian_to_spherical(rq_x, rq_y, rq_z, radius):
    # Calculate φ
    phi = np.arctan2(rq_y, rq_x)
    
    # Calculate r
    r = np.sqrt(rq_x**2 + rq_y**2 + rq_z**2)
    
    # Calculate θ
    theta = np.arccos(rq_z / r)
    
    return theta, phi
