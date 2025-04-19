from scipy.special import lambertw, i0, i1
import math
import numpy as np
import matplotlib.pyplot as plt
import os, sys
simulation_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../', 'Model'))
sys.path.append(simulation_path)
from circle_beam_transmissivity import transmissivity_etab, satellite_ground_distance, beam_waist
from atmospheric_transmissivity import transmissivity_etat, to_decimal_string


a = 0.75  # a      : Aperture of radius (Receiver radis in meters)
a = 0.75                # Aparture radius
G = 6.67430e-11         # Gravitational constant
M_T = 5.972e24          # Earth's mass
D_E = 6378e3            # Earth's radius (km)
h_s = 500e3             # Satellite's altitude
d_o = D_E + h_s
omega = math.sqrt(G * M_T / d_o**3)
T = 2 * math.pi / omega
t = T * 0.0             # Time in orbit of the satellite (the time elapsed when the satellite moves from the reference point)


#=======================================================#
# atmospheric transmissivity(eta_t) parameter
#=======================================================#
    #=====================#
    # distance    : the light-of-sight(LOS) distance(m) between Alice and Bob
    #=====================#
#=======================================================#
# distances = np.arange(100000, 2100000, 100000)
# distances = np.arange(100, 1600, 100) 


#=======================================================#
# Channel loss parameter
#=======================================================#
    #=====================#
    # n_s   : average numher of photon
    # n_D   : dark-current-equivalent averrage photon number
    # eta   : 
    # n_B   : background photons per polarization
    # n_N   : the average number of noise photon reaching each detector
    # gamma : the fraction of transmmited photon
    #======================#
n_s = 100000
n_D = math.pow(10, -6)
eta = 0.5  
n_B = math.pow(10, -3)
n_N = n_B/2 + n_D


def qber(gamma):
    # prob_error = eta * n_N * math.exp(-eta(n_s*gamma+4*n_N))
    prob_error = eta * n_N * math.exp(-eta * (n_s * gamma + 4 * n_N))
    return prob_error

def main():
    displacement = np.arange(0, 3.1, 0.1)
    r = [a * d for d in displacement]

    theta_min = math.radians(0)
    theta_max = math.radians(2)
    theta_list = np.linspace(theta_min, theta_max, 5)

    t_list = [theta / omega for theta in theta_list]

    plt.figure(figsize=(9, 6))

    for t in t_list:
        # 距離 R(t)
        R_t = satellite_ground_distance(h_s, t)
        waist = beam_waist(h_s, t)
        
        # Transmissivity 計算
        eta_b = [transmissivity_etab(a, r_val, waist) for r_val in r]
        eta_t = transmissivity_etat(R_t)

        # gamma = eta_b * eta_t
        gamma = [eta_b_i * eta_t for eta_b_i in eta_b]

        # QBER = qber(gamma)
        qber_values = [qber(gamma_i) for gamma_i in gamma]

        # QBERの値を表示
        theta_deg = omega * t * 180 / math.pi
        print(f"\n--- θ_p = {theta_deg:.2f}° (R={R_t/1e3:.2f} km) ---")
        for i in range(len(displacement)):
            print(f'Gamma: {gamma[i]}')
            print(f"Displacement {displacement[i]:.1f} → QBER = {qber_values[i]:.5e}")

        # グラフ描画
        plt.plot(displacement, qber_values, marker='o', label=f'θ_p={theta_deg:.1f}° (R={R_t/1e3:.1f} km)')

    plt.xlabel(f"Beam centroid displacement,  r / a (a={a} m)", fontsize=14)
    plt.ylabel(r"QBER (Quantum Bit Error Rate)", fontsize=14)
    plt.title("QBER vs Beam Displacement", fontsize=16)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.tight_layout()

    output_path = os.path.join(os.path.dirname(__file__), "qber_distance.png")
    plt.savefig(output_path)
    print(f"✅ Save as: {output_path}")
    plt.show()

if __name__ == "__main__":
    main()