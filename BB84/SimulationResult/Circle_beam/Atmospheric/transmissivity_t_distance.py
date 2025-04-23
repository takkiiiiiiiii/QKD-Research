import matplotlib.pyplot as plt
import numpy as np
import os, sys

simulation_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../', 'Model'))
sys.path.append(simulation_path)
from atmospheric_transmissivity import transmissivity_etat, to_decimal_string


def main():
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