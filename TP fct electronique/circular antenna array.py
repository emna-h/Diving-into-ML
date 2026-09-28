import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from mpl_toolkits.mplot3d import Axes3D

# === Constantes ===
carrierFreq = 2.45e9
c = 3e8
lambda_ = c / carrierFreq
k = 2 * np.pi / lambda_

# === Réseau multi-anneaux ===
rings = 5
radii = np.linspace(0.2 * lambda_, 2.2 * lambda_, rings)
elements_per_ring = [2, 9, 4, 5, 5]
assert len(radii) == len(elements_per_ring)

# === Formation de faisceau ===
theta0deg = 45
theta0 = np.deg2rad(theta0deg)
phi0 = 0

# === 2D azimutal ===
theta = np.linspace(0, 2 * np.pi, 1000)
phi = 0
AF_az = np.zeros_like(theta, dtype=complex)

for ring in range(rings):
    a = radii[ring]
    N = elements_per_ring[ring]
    if N == 0:
        continue
    phi_n = 2 * np.pi * np.arange(N) / N
    for n in range(N):
        phase = k * a * (np.sin(theta) * np.cos(phi - phi_n[n]) - np.sin(theta0) * np.cos(phi0 - phi_n[n]))
        AF_az += np.exp(1j * phase)

AF_norm_az = np.abs(AF_az) / np.max(np.abs(AF_az))
AF_dB_az = 20 * np.log10(AF_norm_az + np.finfo(float).eps)
AF_dB_az[AF_dB_az < -40] = -40
theta_deg = np.rad2deg(theta)

#calcul du gain max non normalisé du lobe principale
AF_abs_az = np.abs(AF_az)
maxVal = np.max(AF_abs_az)
maxVal_non_norm = 20 * np.log10(maxVal + np.finfo(float).eps)
print(f"Gain du lobe principal non normalisé : {maxVal_non_norm:.2f} dB")


# === Tracé polaire ===
plt.figure()
ax = plt.subplot(111, polar=True)
ax.plot(theta, AF_dB_az, linewidth=2)
ax.set_title("Diagramme de rayonnement (coupe en élévation)")
ax.set_rlim([-40, 0])

# === Tracé cartésien ===
plt.figure()
plt.plot(theta_deg, AF_dB_az, linewidth=2)
plt.xlabel("Azimut (°)")
plt.ylabel("Gain (dB)")
plt.title("Diagramme de rayonnement - Azimut")
plt.xlim([0, 360])
plt.ylim([-40, 0])
plt.grid(True)

# === HPBW ===
maxVal_dB = np.max(AF_dB_az)
maxIdx = np.argmax(AF_dB_az)
halfPower = maxVal_dB - 3
AF_dB_ext = np.concatenate((AF_dB_az, AF_dB_az, AF_dB_az))
theta_deg_ext = np.concatenate((theta_deg - 360, theta_deg, theta_deg + 360))
maxIdx_ext = maxIdx + len(theta_deg)

leftIdx_ext = np.where(AF_dB_ext[:maxIdx_ext] <= halfPower)[0][-1] if np.any(AF_dB_ext[:maxIdx_ext] <= halfPower) else None
rightIdx_ext = np.where(AF_dB_ext[maxIdx_ext:] <= halfPower)[0]
rightIdx_ext = rightIdx_ext[0] + maxIdx_ext if len(rightIdx_ext) > 0 else None

if leftIdx_ext is None or rightIdx_ext is None:
    print("Impossible de calculer le HPBW.")
    HPBW = 180
else:
    HPBW = theta_deg_ext[rightIdx_ext] - theta_deg_ext[leftIdx_ext]
    print(f"HPBW = {HPBW:.2f} degrés")

# === Lobe secondaire ===
responseLin = AF_norm_az
peaks, _ = find_peaks(responseLin, distance=5)
pk = responseLin[peaks]
sorted_idx = np.argsort(pk)[::-1]
sorted_pk = pk[sorted_idx]
sorted_angles = theta_deg[peaks][sorted_idx]

threshold_dB = 1
main_lobes_idx = np.where(20 * np.log10(sorted_pk) >= 20 * np.log10(sorted_pk[0]) - threshold_dB)[0]
side_lobe_idx = np.setdiff1d(np.arange(len(sorted_pk)), main_lobes_idx)

if len(side_lobe_idx) == 0:
    print("Aucun lobe secondaire détecté.")
    true_SSL_gain = 0
else:
    true_SSL_gain = 20 * np.log10(sorted_pk[side_lobe_idx[0]])
    true_SSL_angle = sorted_angles[side_lobe_idx[0]]
    #print(f"Gain du lobe principal : {20 * np.log10(sorted_pk[0]):.2f} dB à {sorted_angles[0]:.1f}°")
    print(f"Gain du lobe principal normalisé: {maxVal_dB:.2f} dB")
    print(f"Gain du vrai lobe secondaire : {true_SSL_gain:.2f} dB à {true_SSL_angle:.1f}°")

# === Tracé des pics ===
plt.plot(sorted_angles[main_lobes_idx], 20 * np.log10(sorted_pk[main_lobes_idx]), 'ro', markersize=8)
if len(side_lobe_idx) > 0:
    plt.plot(true_SSL_angle, true_SSL_gain, 'ko', markersize=8)
plt.legend(['Diagramme', 'Lobe(s) principal(aux)', 'Second lobe'])




plt.show()
