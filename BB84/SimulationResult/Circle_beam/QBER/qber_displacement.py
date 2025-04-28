import math
import numpy as np
import matplotlib.pyplot as plt
import os, sys
simulation_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../', 'Model'))
sys.path.append(simulation_path)
from circle_beam_transmissivity import transmissivity_etab, satellite_ground_distance, beam_waist
from atmospheric_transmissivity import transmissivity_etat
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
    # H_a : Receiver's a;titude
    #======================#
a = 0.75               
G = 6.67430e-11       
M_T = 5.972e24      
D_E = 6378e3       
h_s = 500e3       
H_a = 0.01

d_o = D_E + h_s
omega = math.sqrt(G * M_T / d_o**3)
T = 2 * math.pi / omega
t = T * 0.0             # Time in orbit of the satellite (the time elapsed when the satellite moves from the reference point)


#=======================================================#
# QBER parameters
#=======================================================#
    #=====================#
    # n_s   : average numher of photon from Alice
    # e_0   : the error rate of the background
    # Y_0   : the background rate which includes the detector dark count and other background contributions
    # e_dec : the probability that a photon hits the erroneous detector
    #======================#
n_s = 10e8
e_0 = 0.5
Y_0 = 10e-5
e_dec = 0.01




def main():
    r = np.arange(0, 7, 0.5) 

    theta_min = math.radians(0)
    theta_max = math.radians(70)
    theta_list = np.linspace(theta_min, theta_max, 5)

    plt.figure(figsize=(9, 6))

    for theta_zen_rad in theta_list:
        # 距離 R(t)
        L_a = satellite_ground_distance(h_s, H_a, theta_zen_rad)
        waist = beam_waist(h_s, H_a, theta_zen_rad)

        eta_t = transmissivity_etat(tau_zen, theta_zen_rad)

        # Transmissivity 計算
        eta_b = [transmissivity_etab(a, r_val, waist) for r_val in r]

        # gamma = eta_b * eta_t
        gamma = [eta_b_i * eta_t for eta_b_i in eta_b]

        # QBER = qber(gamma)
        qber_values = [qber_loss(gamma_i) * 100 for gamma_i in gamma]

        theta_zen_deg = np.degrees(theta_zen_rad)

        print(f"\n--- θ_p = {theta_zen_deg:.2f}° (R={L_a/1e3:.2f} km, eta_t={eta_t:.4f}) ---")
        for i in range(len(r)):
            print(f'Gamma: {gamma[i]:.5e}, r={r[i]:.1f} m → QBER = {qber_values[i]}')

        plt.plot(r, qber_values, marker='o', label=fr'$\theta_{{zen}}$={theta_zen_deg:.1f}° (R={L_a/1e3:.1f} km)')

    plt.xlabel(f"Beam centroid displacement r (m), (a = {a}m)", fontsize=20)
    plt.ylabel(r"QBER (Quantum Bit Error Rate) [%]", fontsize=20)
    plt.title(fr"QBER vs Beam Displacement  (Clear Sky)", fontsize=20)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.tight_layout()


    output_path = os.path.join(os.path.dirname(__file__), "qber_vs_r_with_tauzen.png")
    plt.savefig(output_path)
    print(f"✅ Save as: {output_path}")
    plt.show()

if __name__ == "__main__":
    main()