import matplotlib.pyplot as plt
import math
import os, sys
import numpy as np
simulation_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../', 'Model'))
sys.path.append(simulation_path)
from circle_beam_transmissivity import transmissivity_etab, satellite_ground_distance, beam_waist, to_decimal_string
from atmospheric_transmissivity import transmissivity_etat


#=======================================================#
# eta_t parameters (Atmospheric loss)
#=======================================================#
    #=====================#
    # tau_zen   : Transmission efficiency at zenith
    # theta_zen : zenith angle
    #======================#
# tau_zen = 0.91


#=======================================================#
# eta_b parameters (Pointing loss)
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

def calculate_theta_deg_from_Rt(R_t):
    cos_theta = (D_E**2 + d_o**2 - R_t**2) / (2 * D_E * d_o)
    
    if abs(cos_theta) > 1:
        raise ValueError("その距離は物理的に実現不可能です。値を確認してください。")

    # θ_p [rad] → [deg]
    theta_p_rad = math.acos(cos_theta)
    theta_p_deg = math.degrees(theta_p_rad)
    
    return theta_p_deg


def main():
    r = np.arange(0, 51, 1)
    tau_zen_list = [0.91, 0.85, 0.75, 0.53]

    # theta_min = 0
    # theta_max = 10
    # theta_list = np.linspace(theta_min, theta_max, 5)

    plt.figure(figsize=(9, 6))

    # for theta_deg in theta_list:
    #     t = math.radians(theta_deg) / omega
    #     waist = beam_waist(h_s, t)
        
    #     eta_b = [transmissivity_etab(a, r_val, waist) for r_val in r]
    #     eta_t = atmospheric_transmittance(tau_zen, theta_deg)

    #     R_t = satellite_ground_distance(h_s, t)

    #     # transmissivity をパーセンテージ（%）に変換
    #     gamma_percent = [eta_b_i * eta_t * 100 for eta_b_i in eta_b]

    #     # グラフ描画
    #     plt.plot(displacement, gamma_percent, marker='o', label=f'θ_p={theta_deg:.1f}° (R(t)={R_t/1e3:.1f} km)')

    waist = 4
    R_t = 500e3
    theta_deg = calculate_theta_deg_from_Rt(R_t)
    for tau_zen in tau_zen_list:
        if tau_zen == 0.91:
            weather = 'Clear sky'
        elif tau_zen == 0.85:
            weather = 'Slightly hazy'
        elif tau_zen == 0.75:
            weather = 'Noticeably hazy'
        else:
            weather = 'Poor visibility'
        eta_b = [transmissivity_etab(a, r_val, waist) for r_val in r]
        eta_t = transmissivity_etat(tau_zen, theta_deg)
        gamma_percent = [eta_b_i * eta_t * 100 for eta_b_i in eta_b]

        plt.plot(r, gamma_percent, marker='o', label=fr'{weather}')

    plt.xlabel(f"Beam centroid displacement r m", fontsize=14)
    plt.ylabel(r"Combined Transmissivity $\gamma$ [%]", fontsize=14)  # % に変更
    plt.title(fr"Received Transmissivity vs Beam Displacement  ($\theta_\text{{zen}}$={theta_deg:.1f}°, LoS={R_t/1e3:.1f} km)", fontsize=16)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.tight_layout()

    output_path = os.path.join(os.path.dirname(__file__), "gamma_vs_displacement_percent.png")
    plt.savefig(output_path)
    print(f"✅ Save as: {output_path}")
    plt.show()

if __name__ == "__main__":
    main()
