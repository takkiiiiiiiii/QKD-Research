import matplotlib.pyplot as plt
import numpy as np
import os, sys
import math

simulation_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../', 'Model'))
sys.path.append(simulation_path)
from atmospheric_transmissivity import transmissivity_etat, to_decimal_string


# Constants
tau_zen = 0.91  # Atmospheric transmissivity at zenith
a = 0.75        # Aperture radius (Receiver)
G = 6.67430e-11 # Gravitational constant
M_T = 5.972e24  # Earth's mass
D_E = 6378e3    # Earth's radius (km)
h_s = 500e3     # Satellite's altitude

d_o = D_E + h_s
omega = math.sqrt(G * M_T / d_o**3)

def main():
    r = np.arange(0, 7, 0.5)  # Beam displacement range [m]
    theta_min = 0
    theta_max = 10
    theta_list = np.linspace(theta_min, theta_max, 5)  # Zenith angles from 0 to 10 degrees

    plt.figure(figsize=(9, 6))

    for theta_deg in theta_list:
        # Calculate eta_t for each zenith angle
        eta_t = [transmissivity_etat(tau_zen, theta_deg) for _ in r]

        # Plot the result
        plt.plot(r, eta_t, marker='o', label=fr'$\theta_{{zen}}$={theta_deg}°')

    plt.xlabel("Beam centroid displacement r (m)", fontsize=20)
    plt.ylabel(r"Atmospheric Transmissivity $\eta_t$", fontsize=20)
    plt.title(f"Atmospheric Transmissivity vs Beam Displacement (Clear Sky)", fontsize=20)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.tight_layout()

    # Save and show the plot
    output_path = "transmissivity_vs_displacement_for_zenith_angle.png"
    plt.savefig(output_path)
    print(f"✅ Save as: {output_path}")
    plt.show()

if __name__ == "__main__":
    main()

if __name__ == '__main__':
    main()