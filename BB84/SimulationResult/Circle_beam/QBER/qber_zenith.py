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
    #======================#
# tau_zen = 0.91 # (Clear Sky)
# tau_zen = 0.85 # (Slightly hazy)
# tau_zen = 0.75 # (Noticeably hazy)
tau_zen = 0.53 # (Poor visibility)

#=======================================================#
# eta_b parameters (Pointing)
#=======================================================#
    #=====================#
    # a       : Aparture radius
    # G       : Gravitational constant
    # M_T     : Earth's mass
    # D_E     : Earth's radius (km)
    # h_s     : Satellite's altitude (m)
    # H_a     : Receiver's altitude
    # theta_d : Divergence angle
    #======================#
a = 0.75               
G = 6.67430e-11       
M_T = 5.972e24      
D_E = 6378e3       
h_s = 500e3       
H_a = 0.01
theta_d_rad = 20e-6


#=======================================================#
# QBER parameters
#=======================================================#
    #=====================#
    # n_s   : average number of photon from Alice
    # e_0   : the error rate of the background
    # Y_0   : the background rate which includes the detector dark count and other background contributions
    # P_AP  : After-pulsing probability
    # e_pol : Probability of the polarisation errors
    #======================#
# n_s = 0.5
# e_0 = 0.5
# Y_0 = 1e-4
# P_AP = 0.02
# e_pol = 0.033


def weather_condition(tau_zen):
    if tau_zen == 0.91:
        weather = 'Clear sky'
    elif tau_zen == 0.85:
        weather = 'Slightly hazy'
    elif tau_zen == 0.75:
        weather = 'Noticeably hazy'
    else:
        weather = 'Poor visibility'
    return weather


def main():
    # displacement r のリスト
    r_list = np.arange(0, 5, 2) 

    # zenith angle θ_zen の範囲 [-60°, 60°] をラジアンで
    theta_min_deg = -60
    theta_max_deg = 60
    theta_zen_deg_list = np.linspace(theta_min_deg, theta_max_deg, 300)  # 滑らかに

    plt.figure(figsize=(9, 6))

    for r in r_list:
        qber_values = []

        for theta_zen_deg in theta_zen_deg_list:
            theta_zen_rad = math.radians(theta_zen_deg)

            # 距離・ビームウエスト
            L_a = satellite_ground_distance(h_s, H_a, theta_zen_rad)
            waist = beam_waist(h_s, H_a, theta_zen_rad, theta_d_rad)

            # 伝搬効率
            eta_t = transmissivity_etat(tau_zen, theta_zen_rad)
            eta_b = transmissivity_etab(a, r, waist)

            gamma = eta_b * eta_t
            qber = qber_loss(gamma) * 100  # [%]

            qber_values.append(qber)

        # プロット
        plt.plot(theta_zen_deg_list, qber_values, label=f"r = {r} m")

    weather = weather_condition(tau_zen)
    plt.xlabel(r"Zenith angle $\theta_{\mathrm{zen}}$ [deg]", fontsize=20)
    plt.ylabel(r"QBER (Quantum Bit Error Rate) [%]", fontsize=20)
    plt.title(fr"QBER vs Beam Displacement  ({weather})", fontsize=20)
    plt.grid(True)
    plt.legend(fontsize=14)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()

    output_path = os.path.join(os.path.dirname(__file__), f'qber_vs_zenith_for_each_r_{weather}.png')
    plt.savefig(output_path)
    print(f"✅ Saved as: {output_path}")
    plt.show()

if __name__ == "__main__":
    main()
