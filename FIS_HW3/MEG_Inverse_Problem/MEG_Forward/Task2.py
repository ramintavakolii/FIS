import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math
from numpy.linalg import matrix_rank
from utility_functions import Calc_G, Conv_coordinates


# -----------------------------  Loading Dipole, MEG sensor and Unit vector Cordinates------------------------------------

data2 = np.load("Dataset/MEG/sensor_coordinates.npz")
data3 = np.load("Dataset/MEG/Unit_Vect_coordinates.npz")

theta = np.radians(45)
phi = np.radians(45)
radius = 0.07

x, y, z = Conv_coordinates(phi, theta, radius)

rq = np.array([x, y, z])
# print(rq)
q = np.array([0, 0, 1])


r_x = data2['x']
r_y = data2['y']
r_z = data2['z']
r = np.concatenate(
    (r_x[:, np.newaxis], r_y[:, np.newaxis], r_z[:, np.newaxis]), axis=1)
# print(r)

er_x = data3['ex']
er_y = data3['ey']
er_z = data3['ez']
er = np.concatenate(
    (er_x[:, np.newaxis], er_y[:, np.newaxis], er_z[:, np.newaxis]), axis=1)
er = np.round(er, 4)
# print(er)

# -----------------------------------------------  Constants ---------------------------------------------------------

m = 33  # Number of MEG Sensor
n = 1   # Number of Diople
R0 = 0.07
R1 = 0.08
R2 = 0.085
R3 = 0.09

sigma = 0.3
sg1 = sigma
sg2 = sigma/80
sg3 = sigma

ep = 8.85 * pow(10, -7)
mu = 4 * math.pi * pow(10, -7)

# --------------------------------------------  Create MEG Laed_Field  -------------------------------------------------------


G = np.zeros((m, 3 * n))

for i in range(m):
    k = 0
    for j in range(n):

        a_ij = Calc_G(er[i], r[i], rq)
        G[i, k] = a_ij[0]
        G[i, k+1] = a_ij[1]
        G[i, k+2] = a_ij[2]
        k = k + 3

G = np.round(G, 8)
# print(G)
Rank = matrix_rank(G)
print("Rank of G = ", Rank)

# -----------------------------  Calculate MEG radial component of magnetic flux density)  ---------------------------------------

B_r = np.dot(G, q)
np.savez("Dataset/MEG/MEG_Lead_Field_2.npz", G=G)
np.savez("Dataset/MEG/MEG_Measurement_Vector.npz", B=B_r)
# print(B_r)

# -------------------------------------  Visiualize MEG sensor ------------------------------------------

fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111, projection='3d')
plt.title('Radial magnetic field for electrodes')

# Set size of each axis
ax.set_box_aspect([1, 1, 1])  # This will make the axes equally spaced
ax.set_xlim([-0.11, 0.11])
ax.set_ylim([-0.11, 0.11])
ax.set_zlim([-0.11, 0.11])

# Set the grid grading for each axis
ax.set_xticks(np.arange(-0.1, 0.1, 0.04))
ax.set_yticks(np.arange(-0.1, 0.1, 0.04))
ax.set_zticks(np.arange(-0.1, 0.1, 0.04))

Bmin = np.min(B_r)
Bmax = np.max(B_r)

# Plotting scatter with actual values
scatter = ax.scatter(r_x, r_y, r_z, c=B_r, cmap='hot',
                     s=50, vmin=Bmin, vmax=Bmax)

# Adding color bar
cbar = plt.colorbar(scatter)
cbar.set_label('|B_r|')

ax.quiver(rq[0], rq[1], rq[2], q[0], q[1], q[2], color='r', length=0.03,
          normalize=True, arrow_length_ratio=0.5)
# -------------------------------------  Plot the hemisphere surface ------------------------------------------

radius = 0.09
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi / 2, 100)

x_hemisphere = radius * np.outer(np.cos(u), np.sin(v))
y_hemisphere = radius * np.outer(np.sin(u), np.sin(v))
z_hemisphere = radius * np.outer(np.ones(np.size(u)), np.cos(v))

ax.plot_surface(x_hemisphere, y_hemisphere, z_hemisphere, color='g', alpha=0.1)

ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')

plt.show()
