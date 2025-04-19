import matplotlib.pyplot as plt
import math
import os, sys
import numpy as np
simulation_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../', 'Model'))
sys.path.append(simulation_path)
from circle_beam_transmissivity import transmissivity_etab, satellite_ground_distance, beam_waist, to_decimal_string

from atmospheric_transmissivity import transmissivity_etat


a = 0.75                # Aparture radius
G = 6.67430e-11         # Gravitational constant
M_T = 5.972e24          # Earth's mass
D_E = 6378e3            # Earth's radius (km)
h_s = 500e3             # Satellite's altitude
d_o = D_E + h_s
omega = math.sqrt(G * M_T / d_o**3)
T = 2 * math.pi / omega
t = T * 0.0           # 周回時間

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

        theta_deg = omega * t * 180 / math.pi
        plt.plot(displacement, gamma, marker='o', label=f'θ_p={theta_deg:.1f}° (R={R_t/1e3:.1f} km)')

    plt.xlabel(f"Beam centroid displacement,  r / a (a={a} m)", fontsize=14)
    plt.ylabel(r"Combined Transmissivity $\gamma = \langle \eta_b \rangle \times \langle \eta_t \rangle$", fontsize=14)
    plt.title("Combined Transmissivity vs Beam Displacement", fontsize=16)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.tight_layout()

    output_path = os.path.join(os.path.dirname(__file__), "gamma_distance.png")
    plt.savefig(output_path)
    print(f"✅ Save as: {output_path}")
    plt.show()

if __name__ == "__main__":
    main()