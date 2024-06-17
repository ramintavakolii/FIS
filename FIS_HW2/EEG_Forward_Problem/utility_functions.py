import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.special import lpmv, lpmn, lpn
from scipy.special import legendre


def cartesian_to_polar(x, y, z):
    r = np.round(np.sqrt(pow(x, 2) + pow(y, 2) + pow(z, 2)),4)
    theta = np.round(np.arccos(z / r),4)
    phi = np.round(np.arctan2(y, x),4)

    return r, theta, phi


def Calc_L(theta, phi, theta_0, phi_0, j, d_n, b, xi, sigma):
    
    P_j1 = lpmv(1, j, np.cos(theta - theta_0))
    P_j = j * legendre(j)(np.cos(theta - theta_0))
    P_j = np.round(P_j, 4)
    P_j1 = np.round(P_j1, 4)
    M = np.array([(P_j1 * np.cos(phi - phi_0)), (P_j1 *
                 np.sin(phi - phi_0)), (P_j * np.cos(theta - theta_0))])
    M = np.round(M,4)
    a_ij = (((2*j + 1) * (pow(b, j-1)) / (j * 4 * np.pi * sigma))
            * ((xi * pow((2*j + 1), 2)) / (d_n * (j + 1)))) * M

    return a_ij


def d_n(n, xi, r_brain, r_skull, r_scalp):

    f1 = r_brain / r_scalp
    f2 = r_skull / r_scalp
    out = ((n + 1) * xi + n) * (((n * xi) / (n + 1)) + 1) + ((1 - xi) * ((n + 1) * xi + n) * (pow(f1,
                                                                                                  (2 * n + 1)) - pow(f2, (2 * n + 1)))) - (n * (pow((1 - xi), 2)) * (pow((f1 / f2), (2 * n + 1))))

    return out
