import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from utility_functions import Conv_coordinates

# -------------------------------------  Calculate Coordinates of MEG sensors ------------------------------------------

num_points = 33

# Define theta and phi values in degrees
theta_degrees = [0] + (np.arange(22.5, 91, 22.5).tolist() * 8)
phi_degrees = [0] * 5 + [45] * 4 + [90] * 4 + [135] * \
    4 + [180] * 4 + [225] * 4 + [270] * 4 + [315] * 4

print(phi_degrees)
print(theta_degrees)

# Convert degrees to radians
theta = np.radians(theta_degrees)
phi = np.radians(phi_degrees)


# Convert spherical coordinates to Cartesian coordinates
radius = 0.09  # radius of the hemisphere
x, y, z = Conv_coordinates(phi, theta, radius)

# Save sorted coordinates to a file
np.savez("Dataset/MEG/sensor_coordinates.npz", x=x, y=y, z=z)


# -------------------------------------  Visualize MEG sensor ------------------------------------------

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title('Visiualize MEG sensor')

# Set size of each axis
ax.set_box_aspect([1, 1, 1])  # This will make the axes equally spaced
ax.set_xlim([-0.11, 0.11])
ax.set_ylim([-0.11, 0.11])
ax.set_zlim([-0.11, 0.11])

# Set the grid grading for each axis
ax.set_xticks(np.arange(-0.1, 0.1, 0.04))
ax.set_yticks(np.arange(-0.1, 0.1, 0.04))
ax.set_zticks(np.arange(-0.1, 0.1, 0.04))


# Scatter plot for MEG sensors with numbers
for i, (xi, yi, zi) in enumerate(zip(x, y, z)):
    ax.scatter(xi, yi, zi, color='b')
    ax.text(xi, yi, zi, f'{i+1}', color='black', fontsize=8)

# -------------------------------------  Plot the hemisphere surface ------------------------------------------

u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi / 2, 100)

x_hemisphere = radius * np.outer(np.cos(u), np.sin(v))
y_hemisphere = radius * np.outer(np.sin(u), np.sin(v))
z_hemisphere = radius * np.outer(np.ones(np.size(u)), np.cos(v))

ax.plot_surface(x_hemisphere, y_hemisphere, z_hemisphere, color='g', alpha=0.4)

ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')

plt.show()
