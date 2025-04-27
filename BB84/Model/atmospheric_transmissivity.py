import math
import matplotlib.pyplot as plt         
import numpy as np

#The atmospheric transmissivity(defined as eta_t) consider absorption and scattering.

#=======================================================#
# atmospheric transmissivity(eta_t) parameter
#=======================================================#
    #=====================#
    # k(kappa): atmospheric absortion coefficient in dB/m
    # d_b     : the light-of-sight(LOS) distance(m) between Alice and Bob
    #=====================#
#=======================================================#



def transmissivity_etat(tau_zen, theta_zen_deg):
    theta_zen_rad = math.radians(theta_zen_deg)
    tau_atm = tau_zen ** (1 / math.cos(theta_zen_rad))
    return tau_atm


def to_decimal_string(x, precision=150):
    if x == 0:
        return "0." + "0" * precision
    return format(x, f'.{precision}f').rstrip('0').rstrip('.')


def main():
    tau_zen_values = [0.91, 0.85, 0.75, 0.53]
    angles = np.linspace(0, 60, 100)

    plt.figure(figsize=(8, 5))

    for tau_zen in tau_zen_values:
        transmissions = [transmissivity_etat(tau_zen, angle) for angle in angles]
        plt.plot(angles, transmissions, label=fr"$\tau_{{zen}} = {tau_zen}$")

    plt.xlabel("Zenith Angle (°)", fontsize=14)
    plt.ylabel("Atmospheric Transmittance", fontsize=14)
    plt.title("Atmospheric Transmittance vs Zenith Angle", fontsize=16)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()


if __name__ == '__main__':
    main()