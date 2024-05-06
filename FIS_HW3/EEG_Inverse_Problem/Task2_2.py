import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy.linalg import matrix_rank
from utility_functions import Conv_coordinates, cartesian_to_spherical
import math
from scipy.interpolate import griddata

# ---------------------------------------- Loading Dipole and EEG_Lead_Field_1 ------------------------------------------------

data1 = np.load("Dataset/EEG/Dipole_coordinates_2.npz")  # Loading Dipole
data2 = np.load("Dataset/EEG/EEG_Lead_Field_2.npz")  # Loading EEG_Lead_Field_1
data3 = np.load("Dataset/EEG/EEG_Measurement_Vector.npz")

rq = data1['rq']
rq_x = rq[104, 0]
rq_y = rq[104, 1]
rq_z = rq[104, 2]

q_0 = np.array([0, 0, 1])
print(q_0.shape)
L = data2['L']
V = data3['V']

Rank = matrix_rank(L)
print("Rank of L = ", Rank)

# ------------------------------------------ calculate Current_source_vector ---------------------------------------------------

q = np.linalg.pinv(L) @ V
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
