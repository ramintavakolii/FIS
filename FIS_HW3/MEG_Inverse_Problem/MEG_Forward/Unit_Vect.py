import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


# -------------------------------------  Calculate Coordinates of Unit Vector ------------------------------------------
data = np.load("sensor_coordinates.npz")
x = data['x']
y = data['y']
z = data['z']

radius = 0.09  # radius of the hemisphere


# Initialize arrays for unit vectors
ex = np.zeros(33)
ey = np.zeros(33)
ez = np.zeros(33)

print(ex.shape)
# Calculate unit vectors
for i in range(33):
    magnitude = np.linalg.norm([x[i], y[i], z[i]])
    ex[i] = x[i] / magnitude
    ey[i] = y[i] / magnitude
    ez[i] = z[i] / magnitude


np.savez("Unit_Vect_coordinates.npz", ex=ex, ey=ey, ez=ez)

# -------------------------------------  Visiualize Unit Vector ---------------------------------------------

fig = plt.figure()
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

ax.quiver(x, y, z, ex, ey, ez, color='r', length=0.015,
          normalize=True, arrow_length_ratio=0.5)
ax.set_title('Visiualize Unit Vector')

# -------------------------------------  Plot the hemisphere surface ------------------------------------------

u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi / 2, 100)

x_hemisphere = radius * np.outer(np.cos(u), np.sin(v))
y_hemisphere = radius * np.outer(np.sin(u), np.sin(v))
z_hemisphere = radius * np.outer(np.ones(np.size(u)), np.cos(v))

ax.plot_surface(x_hemisphere, y_hemisphere, z_hemisphere, color='y', alpha=0.4)

ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')

plt.show()
