import numpy as np
import matplotlib.pyplot as plt
import math
from utility_functions import Calc_G
from numpy.linalg import matrix_rank


# -----------------------------  Loading Dipole, MEG sensor and Unit vector Cordinates------------------------------------

data1 = np.load("Dataset/MEG/Dipole_coordinates_2.npz")
data2 = np.load("Dataset/MEG/sensor_coordinates.npz")
data3 = np.load("Dataset/MEG/Unit_Vect_coordinates.npz")

rq_x = data1['x']
rq_y = data1['y']
rq_z = data1['z']
rq = np.concatenate(
    (rq_x[:, np.newaxis], rq_y[:, np.newaxis], rq_z[:, np.newaxis]), axis=1)
print(rq.shape)


r_x = data2['x']
r_y = data2['y']
r_z = data2['z']
r = np.concatenate(
    (r_x[:, np.newaxis], r_y[:, np.newaxis], r_z[:, np.newaxis]), axis=1)
print(r.shape)

er_x = data3['ex']
er_y = data3['ey']
er_z = data3['ez']
er = np.concatenate(
    (er_x[:, np.newaxis], er_y[:, np.newaxis], er_z[:, np.newaxis]), axis=1)
er = np.round(er, 4)
print(er.shape)

# -----------------------------------------------  Constants ---------------------------------------------------------

m = 33  # Number of MEG Sensor
n = 105
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
        
        a_ij = Calc_G(er[i], r[i], rq[j])
        G[i, k] = a_ij[0]
        G[i, k+1] = a_ij[1]
        G[i, k+2] = a_ij[2]
        k = k + 3
G = np.round(G, 8)
print(G.shape)
Rank = matrix_rank(G)
print("Rank of G = ", Rank)
np.savez("Dataset/MEG/MEG_Lead_Field_1.npz", G=G)
# -------------------------------------  Visiualize G[:, 75] ----------------------------------------------

G_1 = G[:, 74]
print(G_1.shape)

# Creating the plot
plt.figure(figsize=(8, 6))
plt.plot(G_1, label='Column 75 of G')  

plt.xlabel('Sensor Index') 
plt.ylabel('Value of G[:, 74]')  

plt.title('Column 75 of MEG Laed_Field') 
plt.grid(True)
plt.legend()

plt.show()  
