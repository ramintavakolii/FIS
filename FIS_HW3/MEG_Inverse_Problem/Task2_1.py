import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy.linalg import matrix_rank
from utility_functions import sorted, Conv_coordinates
import math

# ---------------------------------------- Loading Dipole and MEG_Lead_Field_1 ------------------------------------------------

data1 = np.load("Dataset/MEG/Dipole_coordinates_2.npz")  # Loading Dipole
data2 = np.load("Dataset/MEG/MEG_Laed_Field_2.npz")  # Loading MEG_Lead_Field_1
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

# ------------------------------------------ calculate Current_source_vector ---------------------------------------------------

q = np.linalg.pinv(G) @ B_r
# q = (np.linalg.inv(G.T@G)@G.T) @ B_r
print('shap of q = ',q.shape)
print('q = ',q)

# calculate Magnitude of each Current_source
norm_q = math.sqrt(q[0]**2 + q[1]**2 + q[2]**2)
print('norm of q = ',norm_q)

# ------------------------------------------ Visiualize Current_source_vector -------------------------------------------------

fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111, projection='3d')
plt.title('Magnitude of Current Sources')       # change

# Set size of each axis
ax.set_box_aspect([1, 1, 1])  # This will make the axes equally spaced
ax.set_xlim([-0.08, 0.08])
ax.set_ylim([-0.08, 0.08])
ax.set_zlim([-0.08, 0.08])

# Set the grid grading for each axis
ax.set_xticks(np.arange(-0.08, 0.08, 0.04))
ax.set_yticks(np.arange(-0.08, 0.08, 0.04))
ax.set_zticks(np.arange(-0.08, 0.08, 0.04))

q_min = np.min(q)
q_max = np.max(q)

# Plotting scatter with actual values
scatter = ax.scatter(rq[104, 0], rq[104, 1], rq[104, 2], c=norm_q, cmap='viridis',
                     s=50, vmin=q_min, vmax=q_max)

# Adding color bar
cbar = plt.colorbar(scatter, pad=0.05)
cbar.set_label('norm_q')  # change


ax.quiver(rq[104, 0], rq[104, 1], rq[104, 2], q[0], q[1], q[2], color='r', length=0.03,
          normalize=True, arrow_length_ratio=0.5)
# -------------------------------------  Plot the hemisphere surface ------------------------------------------

radius = 0.07
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)

x_hemisphere = radius * np.outer(np.cos(u), np.sin(v))
y_hemisphere = radius * np.outer(np.sin(u), np.sin(v))
z_hemisphere = radius * np.outer(np.ones(np.size(u)), np.cos(v))

ax.plot_surface(x_hemisphere, y_hemisphere, z_hemisphere, color='r', alpha=0.1)

ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')

plt.show()
