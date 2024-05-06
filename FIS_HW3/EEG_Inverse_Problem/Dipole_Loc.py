import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from utility_functions import Conv_coordinates


# -------------------------------------  Calculate Coordinates of Diapole  ------------------------------------------

radius = 0.07  # radius of the hemisphere

data1 = np.load("Dataset/EEG/Dipole_coordinates_1.npz")  # Loading Dipole

rq_x = data1['x']
rq_y = data1['y']
rq_z = data1['z']

rq = np.concatenate(
    (rq_x[:, np.newaxis], rq_y[:, np.newaxis], rq_z[:, np.newaxis]), axis=1)


theta = np.radians(45)
phi = np.radians(45)
radius = 0.07  

x_0, y_0, z_0 = Conv_coordinates(phi, theta, radius)
rq_0 = np.array([x_0, y_0, z_0])
q_0 = np.array([0, 0, 1])

rq = np. append(rq, [rq_0], axis=0)

print(rq.shape)

# Save sorted coordinates to a file
np.savez("Dataset/EEG/Dipole_coordinates_2.npz", x=rq[:,0],y=rq[:,1], z=rq[:,2],rq = rq)

# -------------------------------------  Visiualize Diapole  ------------------------------------------

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title('Random Not Reapeted Diapole')

# Set size of each axis
ax.set_xlim([-0.08, 0.08])
ax.set_ylim([-0.08, 0.08])
ax.set_zlim([-0.08, 0.08])

# Set the grid grading for each axis
ax.set_xticks(np.arange(-0.08, 0.08, 0.04))
ax.set_yticks(np.arange(-0.08, 0.08, 0.04))
ax.set_zticks(np.arange(-0.08, 0.08, 0.04))

ax.set_box_aspect([1, 1, 1])  # This will make the axes equally spaced

# Scatter plot for MEG sensors with numbers
for i, (xi, yi, zi) in enumerate(zip(rq[:,0], rq[:,1], rq[:,2])):
    ax.scatter(xi, yi, zi, color='b')
    # ax.text(xi, yi, zi, f'{i+1}', color='black', fontsize=8)

ax.quiver(rq_0[0], rq_0[1], rq_0[2], q_0[0], q_0[1], q_0[2], color='r', length=0.03,
          normalize=True, arrow_length_ratio=0.5)

# -------------------------------------  Plot the hemisphere surface ------------------------------------------

u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)

x_hemisphere = radius * np.outer(np.cos(u), np.sin(v))
y_hemisphere = radius * np.outer(np.sin(u), np.sin(v))
z_hemisphere = radius * np.outer(np.ones(np.size(u)), np.cos(v))

ax.plot_surface(x_hemisphere, y_hemisphere, z_hemisphere, color='r', alpha=0.3)

ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')

plt.show()