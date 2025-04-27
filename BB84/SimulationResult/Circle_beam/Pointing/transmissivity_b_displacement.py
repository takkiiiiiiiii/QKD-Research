import matplotlib.pyplot as plt
import math
import os, sys
import numpy as np
simulation_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../', 'Model'))
sys.path.append(simulation_path)
from circle_beam_transmissivity import transmissivity_etab, beam_waist, satellite_ground_distance


a = 0.75              # Aparture radius
G = 6.67430e-11         # Gravitational constant
M_T = 5.972e24          # Earth's mass
D_E = 6378e3            # Earth's radius (km)
h_s = 500e3             # Satellite's altitude
d_o = D_E + h_s
omega = math.sqrt(G * M_T / d_o**3)
T = 2 * math.pi / omega
t = T * 0.0           # 周回時間

def main():
    # ======= 定義 =========== #
    r = np.arange(0, 7, 0.5)  # ビームのずれ範囲 [m]
    print("===============================")
    print(f'Aperture radius (Receiver): {a} m')


    # θ_pの範囲（ラジアン）
    theta_min = math.radians(0)
    theta_max = math.radians(10)
    theta_list = np.linspace(theta_min, theta_max, 5)  # プロットする角度数（5本）

    plt.figure(figsize=(9, 6))

    for theta_rad in theta_list:
        # 角度→時間へ
        t = theta_rad / omega
        # LoS distance
        R_t = satellite_ground_distance(h_s, t)
        # ビームウエスト計算
        waist = beam_waist(h_s, t)
        # 各rでの eta_b 計算
        eta_b = [transmissivity_etab(a, r_val, waist) for r_val in r]
        # zenith angle [deg]
        theta_deg = math.degrees(theta_rad)

        # プロット
        plt.plot(r, eta_b, marker='o', label=fr'$\theta_{{zen}}$={theta_deg:.1f}° (R={R_t/1e3:.1f} km)')

    print("===============================\n")

    # グラフ装飾
    plt.xlabel(f"Beam centroid displacement r [m] (a={a}m)", fontsize=20)
    plt.ylabel("Transmissivity ⟨η_b⟩", fontsize=20)
    plt.title("Circle Beam Transmissivity vs Beam Displacement", fontsize=20)
    plt.legend(fontsize=12)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.grid(True)
    plt.tight_layout()

    output_path = os.path.join(os.path.dirname(__file__), "transmissivity_vs_displacement_for_zenith_angle.png")
    plt.savefig(output_path)
    print(f"✅ Saved as: {output_path}")

    plt.show()
    print("Simulation Finished!")

if __name__ == "__main__":
    main()