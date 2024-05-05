import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.special import lpmv, lpmn, lpn
from scipy.special import legendre

mu = 4 * math.pi * pow(10, -7)

def cartesian_to_polar(x, y, z):
    r = np.sqrt(pow(x,2) + pow(y,2) + pow(z,2))
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    return r, theta, phi


def Calc_L(theta, phi, theta_0, phi_0, j, d_n, b, xi, sigma):

    P_j1 = lpmv(1, j, np.cos(theta - theta_0))
    P_j = j * legendre(j)(np.cos(theta - theta_0))
    M = np.array([(P_j1 * np.cos(phi - phi_0)), (P_j1 *
                 np.sin(phi - phi_0)), (P_j * np.cos(theta - theta_0))])
    a_ij = (((2*j + 1) * (pow(b, j-1)) / (j * 4 * np.pi * sigma))
            * ((xi * pow((2*j + 1),2)) / (d_n * (j + 1)))) * M

    return a_ij


def d_n(n, xi, r_brain, r_skull, r_scalp):

    f1 = r_brain / r_scalp
    f2 = r_skull / r_scalp
    out = ((n + 1) * xi + n) * (((n * xi) / (n + 1)) + 1) + ((1 - xi) * ((n + 1) * xi + n) * (pow(f1,
                                                                                                (2 * n + 1)) - pow(f2, (2 * n + 1)))) - (n * (pow((1 - xi), 2)) * (pow((f1 / f2), (2 * n + 1))))

    return out

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


