import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy.linalg import matrix_rank
from utility_functions import sorted, Conv_coordinates, cartesian_to_spherical
import math
from scipy.interpolate import griddata

# ---------------------------------------- Loading Dipole and MEG_Lead_Field_1 ------------------------------------------------

data1 = np.load("Dataset/MEG/Dipole_coordinates_2.npz")  # Loading Dipole
data2 = np.load("Dataset/MEG/MEG_Laed_Field_2.npz")  # Loading MEG_Lead_Field_1
data3 = np.load("Dataset/MEG/MEG_Measurement_Vector.npz")

rq = data1['rq']
rq_x = rq[104, 0]
rq_y = rq[104, 1]
rq_z = rq[104, 2]

q_0 = np.array([0, 0, 1])
print(q_0.shape)
G = data2['G']
B_r = data3['B']

Rank = matrix_rank(G)
print("Rank of G = ", Rank)

# ------------------------------------------ calculate Current_source_vector ---------------------------------------------------

q = np.linalg.pinv(G) @ B_r
print('shap of q = ', q.shape)

# calculate Magnitude of each Current_source
norm_q = math.sqrt(q[0]**2 + q[1]**2 + q[2]**2)
print('norm of q = ', norm_q)
#  -------------------------------- Convert spherical coordinates to Cartesian coordinates -------------------------------------

# Compute spherical coordinates
theta, phi = cartesian_to_spherical(rq_x, rq_y, rq_z, 0.07)

# convert radian to degree
phi_rad = np.rad2deg(phi)
theta_rad = np.rad2deg(theta)

#  --------------------------------------------- Calculate the relative error -------------------------------------------------

# Calculate relative_q0_error
relative_q0_error = np.sqrt(
    np.sum((q - q_0)**2, axis=0)) / np.sqrt(np.sum(q_0**2, axis=0))
print('The relative q0 error =', relative_q0_error)

# Calculate relative_q_error
relative_q_error = np.sqrt(
    np.sum((q - q_0)**2, axis=0)) / np.sqrt(np.sum(q_0**2, axis=0))
print('The relative q error =', relative_q_error)
