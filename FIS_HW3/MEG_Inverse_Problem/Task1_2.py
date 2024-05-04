import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy.linalg import matrix_rank
from utility_functions import sorted, Conv_coordinates, cartesian_to_spherical
import math
from scipy.interpolate import griddata

# ------------------------------------------- Loading Dipole and MEG_Lead_Field_1 ------------------------------------------------

data1 = np.load("Dataset/MEG/Dipole_coordinates_2.npz")  # Loading Dipole
data2 = np.load("Dataset/MEG/MEG_Lead_Field_1.npz")  # Loading MEG_Lead_Field_1
data3 = np.load("Dataset/MEG/MEG_Measurement_Vector.npz")

rq_x = data1['x']
rq_y = data1['y']
rq_z = data1['z']

rq = data1['rq']
q_0 = np.array([0, 0, 1])

G = data2['G']
B_r = data3['B']

Rank = matrix_rank(G)
print("Rank of G = ", Rank)

# -------------------------------------------- calculate Current_source_vector ---------------------------------------------------

q = G.T @ np.linalg.inv(G@G.T) @ B_r
# q = np.linalg.pinv(G) @ B_r
norm_q = np.array(np.zeros(105,))

# calculate Magnitude of each Current_source
for i in range(0, 105):
    j = 3*i
    norm_q[i] = math.sqrt(q[0+j]**2 + q[1+j]**2 + q[2+j]**2)


#  -------------------------------- Convert spherical coordinates to Cartesian coordinates -------------------------------------

# Compute spherical coordinates
theta, phi = cartesian_to_spherical(rq_x, rq_y, rq_z, 0.07)

# convert radian to degree
phi_rad = np.rad2deg(phi)
theta_rad = np.rad2deg(theta)

#  --------------------------------------------- Calculate the relative error -------------------------------------------------
q1 = np.reshape(q, (105, 3))
# Calculate relative_q0_error
relative_q0_error = np.sqrt(
    np.sum((q1[104, :] - q_0)**2, axis=0)) / np.sqrt(np.sum(q_0**2, axis=0))
print('The relative q0 error =', relative_q0_error)

# Calculate relative_q_error
relative_q_error = np.sqrt(np.sum(
    np.mean((q1 - q_0), axis=0)**2, axis=0)) / np.sqrt(np.sum(q_0**2, axis=0))
print('The relative q error =', relative_q_error)

# -------------------------------------------------- Plotting a 3D Meshgrid --------------------------------------------------

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Setting the title and labels for the plot
plt.title('Magnitude of Current Sources', c='r', fontsize=16)
ax.set_xlabel('phi (degree)', c='b')
ax.set_ylabel('theta (degree)', c='b')
ax.set_zlabel('q1_norm', c='b')

# Creating a meshgrid for phi and theta
mesh_phi, mesh_theta = np.meshgrid(phi_rad, theta_rad)

# Increasing the number of points in the meshgrid for smoother surface
phi_rad_dense = np.linspace(phi_rad.min(), phi_rad.max(), 360)
theta_rad_dense = np.linspace(theta_rad.min(), theta_rad.max(), 360)
mesh_phi_dense, mesh_theta_dense = np.meshgrid(phi_rad_dense, theta_rad_dense)

# Performing interpolation on denser grid
interpolated_q1_norm = griddata(
    (phi_rad, theta_rad), norm_q, (mesh_phi_dense, mesh_theta_dense), method='linear')

# Plotting the surface
surf = ax.plot_surface(mesh_phi_dense, mesh_theta_dense,
                       interpolated_q1_norm, cmap='viridis', edgecolor='none')

plt.show()
