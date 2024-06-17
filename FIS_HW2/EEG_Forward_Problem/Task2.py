import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math
from utility_functions import Calc_L,d_n, cartesian_to_polar
from numpy.linalg import matrix_rank


# --------------------------------  Loading Dipole, EEG sensor Cordinates -----------------------------------------


data2 = np.load("sensor_coordinates.npz")

theta_0 = np.radians(45)
phi_0 = np.radians(45)
radius = 0.07  

x = radius * np.sin(theta_0) * np.cos(phi_0)
y = radius * np.sin(theta_0) * np.sin(phi_0)
z = radius * np.cos(theta_0)

q = np.array([0, 0, 1])
rq = np.array([x, y, z])

r_x = data2['x']
r_y = data2['y']
r_z = data2['z']

r, theta, phi = cartesian_to_polar(r_x, r_y, r_z)

print(phi.shape)
print(theta.shape)


# -----------------------------------------------  Constants ---------------------------------------------------------

m = 33       # Number of EEG Sensor
n = 1      # Number of Dipole
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

# --------------------------------------------  Create EEG Laed_Field  -------------------------------------------------------

L = np.zeros((m, 3 * n))

xi = sg2/sigma
b = R0/R3

# m = Number of EEG Sensor 
for i in range(m):
    k = 0
    dn = d_n(1, xi, R1, R2, R3)
    a_ij = Calc_L(theta[i], phi[i], theta_0, phi_0, 1, dn, b, xi, sigma)
        
    L[i, k] = a_ij[0]
    L[i, k+1] = a_ij[1]
    L[i, k+2] = a_ij[2]
    

print(L.shape)
Rank = matrix_rank(L)
print("Shape of L = ", L.shape)
print("Rank of L = ", Rank)


# ---------------------------------  Calculate EEG Voltage of sensors  ---------------------------------------

V = np.dot(L, q)
np.savez("EEG_Laed_Field.npz", L = L)
print(V.shape)

# -------------------------------------  Visiualize EEG sensor ------------------------------------------

fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111, projection='3d')

# Set size of each axis
ax.set_box_aspect([1, 1, 1])  # This will make the axes equally spaced
ax.set_xlim([-0.11, 0.11])
ax.set_ylim([-0.11, 0.11])
ax.set_zlim([-0.11, 0.11])

# Set the grid grading for each axis
ax.set_xticks(np.arange(-0.1, 0.1, 0.04))
ax.set_yticks(np.arange(-0.1, 0.1, 0.04))
ax.set_zticks(np.arange(-0.1, 0.1, 0.04))
plt.title('Voltage value of sensors') 

vmin = np.min(V)
vmax = np.max(V)

# Plotting scatter with actual values
scatter = ax.scatter(r_x, r_y, r_z, c=V, cmap='hot', s=50, vmin=vmin, vmax=vmax)

# Adding color bar
cbar = plt.colorbar(scatter)
cbar.set_label('|V|')

for i in range(m):
    ax.text(r_x[i], r_y[i], r_z[i], f'{i+1}', color='black', fontsize=8)

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
