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
    # H_a : Receiver's altitude
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


def main():
    r = np.arange(0, 7, 0.5)

    # --- θ_p の範囲設定 ---
    theta_min = math.radians(0)     # 0°
    theta_max = math.radians(60)    # 10°
    theta_list = np.linspace(theta_min, theta_max, 7)
    # t_list = [theta / omega for theta in theta_list]

    # --- 大気損失設定（例としてtau_zen = 0.91だけ使用） ---
    tau_zen = 0.91

    # --- プロット ---
    plt.figure(figsize=(9, 6))

    for theta_zen_rad in theta_list:
        # 衛星-地上間距離
        L_a = satellite_ground_distance(h_s, H_a, theta_zen_rad)
        # ビームウエスト計算
        waist = beam_waist(h_s, H_a, theta_zen_rad)
        # 現在の時刻tにおける天頂角 [deg]
        # theta_zen_deg = omega * t * 180 / math.pi

        # 大気損失
        eta_t = transmissivity_etat(tau_zen, theta_zen_rad)

        # ビーム拡がりロス
        eta_b = [transmissivity_etab(a, r_val, waist) for r_val in r]

        # 総合損失
        gamma = [eta_b_i * eta_t * 100 for eta_b_i in eta_b]


        theta_zen_deg = math.degrees(theta_zen_rad)
        
        print(f"\n--- θ_p = {theta_zen_deg:.2f}° (R={L_a/1e3:.2f} km) ---")
       
        plt.plot(r, gamma, marker='o', label=fr'$\theta_{{zen}}$={theta_zen_deg:.1f}° (R={L_a/1e3:.1f} km)')

    # --- グラフ仕上げ ---
    plt.xlabel(r"Beam centroid displacement $r$ [m]", fontsize=20)
    plt.ylabel(r"Combined Transmissivity $\gamma$ [%]", fontsize=20)  # % に変更
    plt.title(r"$\gamma$ vs Beam Displacement (Clear Sky)", fontsize=20)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.tight_layout()

    # --- 保存 & 表示 ---
    output_path = os.path.join(os.path.dirname(__file__), "gamma_vs_displacement_plot.png")
    plt.savefig(output_path)
    print(f"✅ Saved as: {output_path}")
    plt.show()

if __name__ == '__main__':
    main()