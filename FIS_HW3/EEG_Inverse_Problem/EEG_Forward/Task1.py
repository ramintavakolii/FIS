import numpy as np
import matplotlib.pyplot as plt
import math
from utility_functions import Calc_L,d_n, cartesian_to_polar
from numpy.linalg import matrix_rank


# --------------------------------  Loading Dipole, EEG sensor Cordinates -----------------------------------------

data1 = np.load("Dataset/EEG/Dipole_coordinates_2.npz")
data2 = np.load("Dataset/EEG/sensor_coordinates.npz")

rq_x = data1['x']
rq_y = data1['y']
rq_z = data1['z']

r_0, theta_0, phi_0 = cartesian_to_polar(rq_x, rq_y, rq_z)

print(phi_0.shape)
print(theta_0.shape)


r_x = data2['x']
r_y = data2['y']
r_z = data2['z']

r, theta, phi = cartesian_to_polar(r_x, r_y, r_z)

print(phi.shape)
print(theta.shape)


# -----------------------------------------------  Constants ---------------------------------------------------------

m = 33       # Number of EEG Sensor
n = 105      # Number of Dipole
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
print(phi)
# m = Number of EEG Sensor      n =  Number of Dipole
for i in range(m):
    k = 0
    for j in range(n):

        dn = d_n(j+1, xi, R1, R2, R3)
        dn = np.round(dn,2)
        a_ij = Calc_L(theta[i], phi[i], theta_0[j], phi_0[j], j+1, dn, b, xi, sigma)
        
        L[i, k] = a_ij[0]
        L[i, k+1] = a_ij[1]
        L[i, k+2] = a_ij[2]
        k = k + 3

print(L)

Rank = matrix_rank(L)
print("Shape of L = ", L.shape)
print("Rank of L = ", Rank)
np.savez("Dataset/EEG/EEG_Lead_Field_1.npz", L = L)
# ----------------------------------------------  Visiualize L[:, 74] ---------------------------------------------------

G_1 = L[:, 74]
print(G_1.shape)

# Creating the plot
plt.figure(figsize=(10, 7))
plt.plot(G_1, label='Column 75 of G')

plt.xlabel('Sensor Index')
plt.ylabel('Value of G[:, 74]')

plt.title('Column 75 of MEG Laed_Field')
plt.grid(True)
plt.legend(loc='upper right', fontsize='large', fancybox=True, frameon=True, facecolor='lightgray', edgecolor='red')


plt.show()
