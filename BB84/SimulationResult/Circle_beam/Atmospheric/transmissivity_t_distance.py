import matplotlib.pyplot as plt
import numpy as np
import os, sys

simulation_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../', 'Model'))
sys.path.append(simulation_path)
from atmospheric_transmissivity import transmissivity_etat, to_decimal_string


#The atmospheric transmissivity(defined as eta_t) consider absorption and scattering.

#=======================================================#
# atmospheric transmissivity(eta_t) parameter
#=======================================================#
    #=====================#
    # k(kappa): atmospheric absortion coefficient in dB/m
    # d_b     : the light-of-sight(LOS) distance(m) between Alice and Bob (R(t))
    #=====================#
#=======================================================#



# def transmissivity_etat(d_b):
#     kappa = 0.43 * pow(10, -3)
#     eta_t = pow(10, -0.1*kappa*d_b)
#     return eta_t

# def to_decimal_string(x, precision=150):
#     if x == 0:
#         return "0." + "0" * precision
#     return format(x, f'.{precision}f').rstrip('0').rstrip('.')

# def main():
#     # distances = np.arange(100, 1600, 100) 
#     # Starlink uses Low-Earth Orbit (LEO) satellites to orbit the Earth at a height of between 180 to 2,000km.
#     distances = np.arange(100000, 2100000, 100000) 
#     all_eta_t = [transmissivity_etat(d_b) for d_b in distances]
#     for j, eta_t in enumerate(all_eta_t):
#         print(f"d_b = {distances[j]} → <ηt> = {to_decimal_string(eta_t)}")
#         print("===============================\n")
#     print("Simulation Finish !!\n")

def main():
    # 距離（m単位）
    distances = np.arange(500e3, 555e3, 5e3)  # 500km～550km
    all_eta_t = [transmissivity_etat(d_b) for d_b in distances]

    # 数値の確認用出力
    for j, eta_t in enumerate(all_eta_t):
        print(f"d_b = {distances[j]} m → <ηt> = {to_decimal_string(eta_t)}")
        print("===============================\n")

    print("Simulation Finish !!\n")

    # グラフ描画
    plt.figure(figsize=(8, 5))
    plt.plot(distances / 1000, all_eta_t, marker='o', linestyle='-', color='b', label=r'$\langle \eta_t \rangle$')

    # ラベルやタイトル
    plt.xlabel("Distance $d_b$ (km)", fontsize=14)
    plt.ylabel(r"Transmissivity $\langle \eta_t \rangle$", fontsize=14)
    plt.title("Transmissivity vs Distance", fontsize=16)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(__file__), "transmissivity_t_distance_plot.png")
    plt.savefig(output_path)
    print(f"✅ Save as: {output_path}")
    plt.show()

if __name__ == '__main__':
    main()