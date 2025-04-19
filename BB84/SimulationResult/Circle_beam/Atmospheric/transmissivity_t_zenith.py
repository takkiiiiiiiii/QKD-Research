import math
import matplotlib.pyplot as plt         
import numpy as np
import os, sys
simulation_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../', 'Model'))
sys.path.append(simulation_path)
from atmospheric_transmissivity import atmospheric_transmittance



def main():
    tau_zen_values = [0.91, 0.85, 0.75, 0.53]
    angles = np.linspace(0, 60, 100)

    plt.figure(figsize=(8, 5))

    for tau_zen in tau_zen_values:
        transmissions = [atmospheric_transmittance(tau_zen, angle) for angle in angles]
        plt.plot(angles, transmissions, label=fr"$\tau_{{zen}} = {tau_zen}$")

    plt.xlabel("Zenith Angle (°)", fontsize=14)
    plt.ylabel("Atmospheric Transmittance", fontsize=14)
    plt.title("Atmospheric Transmittance vs Zenith Angle", fontsize=16)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(__file__), "transmissivity_t_zenith_plot.png")
    plt.savefig(output_path)
    print(f"✅ Save as: {output_path}")
    plt.show()

if __name__ == "__main__":
    main()


if __name__ == '__main__':
    main()