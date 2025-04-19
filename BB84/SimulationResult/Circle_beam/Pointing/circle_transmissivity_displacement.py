import matplotlib.pyplot as plt
import math
import os, sys
import numpy as np
simulation_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../', 'Model'))
sys.path.append(simulation_path)
from circle_beam_transmissivity import transmissivity_etab, beam_waist, to_decimal_string


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
    # =======Definition of parameter =========== #
    displacement = np.arange(0, 3.1, 0.1)
    r = [a*d for d in displacement]
    print("===============================")
    print(f'Aperture of radius (Receiver radis in meters): {a} m')

    # range of θ_p
    theta_min = -math.radians(0)
    theta_max = math.radians(10)

    theta_list = np.linspace(theta_min, theta_max, 5)

    # それを t に変換
    t_list = [theta / omega for theta in theta_list]
    for t in t_list:
        waist = beam_waist(h_s, t)
        eta_b = [transmissivity_etab(a, r_val, waist) for r_val in r]
        theta_deg = omega * t * 180 / math.pi
        # if theta_deg != 0:
        plt.plot(displacement, eta_b, marker='o', label=f'Angular position(θ_p)={theta_deg:.1f}°')
    # t = T * 0.0
    # waist = beam_waist(h_s, t)
    # eta_b = [transmissivity_etab(a, r, waist) for r in r]
    # plt.plot(displacement, eta_b, marker='o', label=f'Beam waist={waist:.2f}m')
    print("===============================\n")



    plt.xlabel(f"Beam centroid displacement,  r / a (a={a}m)", fontsize=14)
    plt.ylabel("Transmissivity ⟨η_b⟩", fontsize=14)
    plt.title("Circle Beam Transmissivity vs r/a", fontsize=16)
    plt.legend(fontsize=20)
    plt.grid(True)
    plt.tight_layout()


    output_path = os.path.join(os.path.dirname(__file__), "circle_transmissivity_displacement_plot.png")
    plt.savefig(output_path)
    print(f"✅ Save as: {output_path}")

    plt.show()
    print("Simulation Finish !!")

if __name__ == "__main__":
    main()