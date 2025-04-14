import matplotlib.pyplot as plt
import math
import os, sys
import numpy as np
simulation_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../', 'Model'))
sys.path.append(simulation_path)
from circle_beam_transmissivity import transmissivity_etab, to_decimal_string


D_r = 0.35

def main():
    # =======Definition of parameter =========== #
    # ratios = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    a = D_r/2
    # displacement = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    displacement = np.arange(0, 3.1, 0.1)
    r = [a*d for d in displacement]
    W = [0.2*a, 1.0*a, 1.8*a]
    W_show = [0.2, 1.0, 1.8]
    print("===============================")
    print(f'Aperture of radius (Receiver radis in meters): {a} m')
    for i, w_1 in enumerate(W):
        print(f'Beam spot radius: {W[i]}')
        eta_b = [transmissivity_etab(a, r, W[i]) for r in r]
        plt.plot(displacement, eta_b, marker='o', label=f'{W_show[i]}a')

    print("===============================\n")


    plt.xlabel(f"Beam cenroid displacement,  r / a (a={a}m)")
    plt.ylabel("Transmissivity ⟨η_b⟩")
    plt.title("Circle Beam Transmissivity vs r/a")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # 画像ファイルとして保存
    output_path = os.path.join(os.path.dirname(__file__), "circle_transmissivity_plot.png")
    plt.savefig(output_path)
    print(f"✅ Save as: {output_path}")

    plt.show()
    print("Simulation Finish !!")

if __name__ == "__main__":
    main()