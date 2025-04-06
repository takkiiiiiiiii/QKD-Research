import matplotlib.pyplot as plt
import math
import os, sys
import numpy as np
simulation_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Model'))
sys.path.append(simulation_path)
from receiver_efficiency import transmissivity, to_decimal_string


D_r = 0.35
a = D_r / 2

def main():
    # =======Definition of parameter =========== #
    # ratios = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    ratios = np.arange(0, 3.1, 0.1)
    r0 = [r * a for r in ratios]
    mag_w1 = [0.2, 1.0, 1.8]
    mag_w2 = [0.1, 0.9, 1.7]
    chi = [math.pi/3, math.pi/4, math.pi/5]
    chi_show = [3, 4, 5]

    print("===============================")
    print(f'Aperture of radius (Receiver radius in meters): {a} m')
    
    for i, w_1 in enumerate(mag_w1):
        print(f'Long axis: {mag_w1[i]} * {a}')
        print(f'Short axis: {mag_w2[i]} * {a}')
        print(f'Chi: π / {chi_show[i]}')
        
        beam_centroid_displacement = [r / a for r in r0]
        eta_b = [transmissivity(b, chi[i], mag_w1[i]*a, mag_w2[i]*a) for b in beam_centroid_displacement]

        # print("Transmissivity values:")
        # for j, eta in enumerate(eta_b):
            # print(f"  r0/a = {ratios[j]} → <ηb> = {to_decimal_string(eta)}")
        
        # グラフにプロット
        plt.plot(ratios, eta_b, marker='o', label=f'χ = π/{chi_show[i]}')

        print("===============================\n")

    # グラフ設定
    plt.xlabel(f"Beam cenroid displacement,  r₀ / a (a={a}m)")
    plt.ylabel("Transmissivity ⟨η_b⟩")
    plt.title("Transmissivity vs r₀/a")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # 画像ファイルとして保存
    output_path = os.path.join(os.path.dirname(__file__), "transmissivity_plot.png")
    plt.savefig(output_path)
    print(f"✅ Save: {output_path}")

    plt.show()
    print("Simulation Finish !!")

if __name__ == "__main__":
    main()