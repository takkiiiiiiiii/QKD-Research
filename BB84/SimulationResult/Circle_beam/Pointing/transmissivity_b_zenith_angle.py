import matplotlib.pyplot as plt
import math
import os, sys
import numpy as np

# === モジュールパスの設定 === #
simulation_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../', 'Model'))
sys.path.append(simulation_path)
from circle_beam_transmissivity import transmissivity_etab, beam_waist

# === 定数 === #
a = 0.75                # Aperture radius
G = 6.67430e-11         # Gravitational constant
M_T = 5.972e24          # Earth's mass
D_E = 6378e3            # Earth's radius (m)
h_s = 500e3             # Satellite altitude (m)
d_o = D_E + h_s         # Distance from Earth center to satellite
omega = math.sqrt(G * M_T / d_o**3)
T = 2 * math.pi / omega

# === ビームのズレ（r）のリスト === #
r_fixed_list = [10]

def main():
    # 天頂角θの範囲 [°] をラジアンに変換
    theta_min = math.radians(-60)
    theta_max = math.radians(60)
    theta_rad_list = np.linspace(theta_min, theta_max, 30)  # プロットする角度数（5本）
    # theta_deg_list = np.linspace(-60, 60, 100)
    theta_deg_list = np.degrees(theta_rad_list)

    plt.figure(figsize=(9, 6))

    for r_fixed in r_fixed_list:
        eta_b_list = []
        for theta_zen_rad in theta_rad_list:
            waist = beam_waist(h_s, theta_zen_rad)
            eta_b = transmissivity_etab(a, r_fixed, waist)
            eta_b_list.append(eta_b)

        # プロット
        plt.plot(theta_deg_list, eta_b_list, label=f"r = {r_fixed} m", marker='o')

    # グラフ装飾
    plt.xlabel(fr'Zenith angle ($\theta_{{zen}}$) [deg]', fontsize=20)
    plt.ylabel("Transmissivity ⟨η_b⟩", fontsize=20)
    plt.title(r'Beam Transmissivity vs Zenith Angle ($\theta_{\rm zen}$)', fontsize=20)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)

    plt.tick_params(axis='both', which='major', length=8, width=1.5)
    plt.tick_params(axis='both', which='minor', length=4, width=1)

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # 保存
    output_path = os.path.join(os.path.dirname(__file__), "transmissivity_vs_zenith_angle_multiple_r.png")
    plt.savefig(output_path)
    print(f"✅ Saved as: {output_path}")

    plt.show()
    print("Simulation Finished!")

if __name__ == "__main__":
    main()
