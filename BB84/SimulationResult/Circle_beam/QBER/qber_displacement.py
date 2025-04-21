from scipy.special import lambertw, i0, i1
import math
import numpy as np
import matplotlib.pyplot as plt
import os, sys
simulation_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../', 'Model'))
sys.path.append(simulation_path)
from circle_beam_transmissivity import transmissivity_etab, satellite_ground_distance, beam_waist
from atmospheric_transmissivity import atmospheric_transmittance
from qber import qber_loss



#=======================================================#
# eta_t parameters (Atmospheric loss)
#=======================================================#
    #=====================#
    # tau_zen   : Transmission efficiency at zenith
    # theta_zen : zenith angle
    #======================#
tau_zen = 0.91


#=======================================================#
# eta_b parameters (Pointing)
#=======================================================#
    #=====================#
    # a   : Aparture radius
    # G   : Gravitational constant
    # M_T : Earth's mass
    # D_E : Earth's radius (km)
    # h_s : Satellite's altitude
    #======================#
a = 0.75               
G = 6.67430e-11       
M_T = 5.972e24      
D_E = 6378e3       
h_s = 500e3       

d_o = D_E + h_s
omega = math.sqrt(G * M_T / d_o**3)
T = 2 * math.pi / omega
t = T * 0.0             # Time in orbit of the satellite (the time elapsed when the satellite moves from the reference point)


#=======================================================#
# QBER parameters
#=======================================================#
    #=====================#
    # n_s   : average numher of photon
    # n_D   : dark-current-equivalent averrage photon number
    # eta   : APD (single-photon avalanche photodiode) detection efficiency
    # n_B   : background photons per polarization
    # n_N   : the average number of noise photon reaching each detector
    # gamma : the fraction of transmmited photon
    #======================#
n_s = 10e8
n_D = math.pow(10, -6)
eta = 0.5  
n_B = math.pow(10, -3)
n_N = n_B/2 + n_D


def main():
    r = np.arange(0, 51, 1) 

    theta_min = math.radians(0)
    theta_max = math.radians(10)
    theta_list = np.linspace(theta_min, theta_max, 5)

    t_list = [theta / omega for theta in theta_list]

    plt.figure(figsize=(9, 6))

    for t in t_list:
        # 距離 R(t)
        R_t = satellite_ground_distance(h_s, t)
        waist = beam_waist(h_s, t)

        theta_deg = omega * t * 180 / math.pi
        eta_t = atmospheric_transmittance(tau_zen, theta_deg)

        # Transmissivity 計算
        eta_b = [transmissivity_etab(a, r_val, waist) for r_val in r]

        # gamma = eta_b * eta_t
        gamma = [eta_b_i * eta_t for eta_b_i in eta_b]

        # QBER = qber(gamma)
        qber_values = [qber_loss(eta, n_s, n_N, gamma_i) for gamma_i in gamma]

        # QBERの値を表示
        print(f"\n--- θ_p = {theta_deg:.2f}° (R={R_t/1e3:.2f} km, eta_t={eta_t:.4f}) ---")
        for i in range(len(r)):
            print(f'Gamma: {gamma[i]:.5e}, r={r[i]:.1f} m → QBER = {qber_values[i]}')

        # グラフ描画
        plt.plot(r, qber_values, marker='o', label=fr'$\theta_{{zen}}$={theta_deg:.1f}° (R={R_t/1e3:.1f} km)')

    plt.xlabel(f"Beam centroid displacement r (m)", fontsize=14)
    plt.ylabel(r"QBER (Quantum Bit Error Rate)", fontsize=14)
    plt.title(fr"QBER vs Beam Displacement  ($\tau_{{zen}}$={tau_zen})", fontsize=16)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.tight_layout()

    output_path = os.path.join(os.path.dirname(__file__), "qber_vs_r_with_tauzen.png")
    plt.savefig(output_path)
    print(f"✅ Save as: {output_path}")
    plt.show()

if __name__ == "__main__":
    main()