import math
import matplotlib.pyplot as plt         
import numpy as np
import os, sys
simulation_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../', 'Model'))
sys.path.append(simulation_path)
from atmospheric_transmissivity import transmissivity_etat



def main():
    tau_zen_values = [0.91, 0.85, 0.75, 0.53]
    angles = np.linspace(-60, 60, 100)

    plt.figure(figsize=(8, 5))

    for tau_zen in tau_zen_values:
        if tau_zen == 0.91:
            weather = fr'Clear sky($\tau_{{zen}}$={tau_zen})'
        elif tau_zen == 0.85:
            weather = fr'Slightly hazy ($\tau_{{zen}}$={tau_zen})'
        elif tau_zen == 0.75:
            weather = fr'Noticeably hazy ($\tau_{{zen}}$={tau_zen})'
        else:
            weather = fr'Poor visibility ($\tau_{{zen}}$={tau_zen})'
        transmissions = [transmissivity_etat(tau_zen, angle) for angle in angles]
        plt.plot(angles, transmissions, label=fr"{weather}")

    plt.xlabel(fr"Zenith Angle ($\theta_{{zen}}$)", fontsize=14)
    plt.ylabel("Atmospheric Transmittance", fontsize=14)
    plt.title(fr"Atmospheric Transmittance vs Zenith Angle($\theta_{{zen}}$)", fontsize=16)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(__file__), "transmissivity_t_zenith_plot.png")
    plt.savefig(output_path)
    print(f"✅ Save as: {output_path}")
    plt.show()

if __name__ == "__main__":
    main()
