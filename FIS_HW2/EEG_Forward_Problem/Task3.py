import numpy as np
import matplotlib.pyplot as plt


# ----------------------------------------------- Calculate W(t) -------------------------------------------------

def w(t):
    return 120 * np.sin(8 * np.pi * t) + 45 * np.sin(14 * np.pi * t) + 30 * np.sin(20 * np.pi * t) + 15 * np.sin(40 * np.pi * t) + 5 * np.sin(80 * np.pi * t)


# Set Frequency to 1000
t = np.linspace(0, 2, 1000)

# Calculate w(t) for each t
w_values = w(t)


data = np.load("EEG_Laed_Field.npz")
L = data['L']
q = np.array([0, 0, 1])[:, np.newaxis]  # Reshape q to (3, 1)
q1 = np.array([0, 0, 1])[:, np.newaxis]  # Reshape q to (3, 1)

print(L[29].shape)
print(w_values.shape)

V = L[29] @ q
V_new = L[29] @ (q1 * w_values)


# -------------------------------------  Visiualize EEG Sensor Number 30 ------------------------------------------
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(13, 7))

# Plot w(t)
ax1.plot(t, w_values, label=r'w(t)', color='r')
ax1.set_title('Plot of w(t)')
ax1.set_xlabel('t')
ax1.set_ylabel('w(t)')
ax1.legend()
ax1.grid(True)


# Plot V
ax2.plot(t, np.full(t.shape, (V)), label=r'V')
ax2.set_title('Constant Sensor Number 30')
ax2.set_xlabel('Time(s)')
ax2.set_ylabel('Voltage')
ax2.legend()
ax2.grid(True)


# Plot V_new
ax3.plot(t, V_new, label=r'V_new')
ax3.set_title('Variable Sensor Number 30')
ax3.set_xlabel('Time(s)')
ax3.set_ylabel('Voltage')
ax3.legend()
ax3.grid(True)

plt.tight_layout()
plt.show()
