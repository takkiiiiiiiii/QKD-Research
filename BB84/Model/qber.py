from scipy.special import lambertw, i0, i1
import math
import numpy as np
# from Model.eliptic_beam_transmissivity import transmissivity, to_decimal_string


D_r = 0.35 # D_r    : Deceiver diameter in meters
a = D_r/2  # a      : Aperture of radius (Receiver radis in meters)
ratios = np.arange(0, 3.1, 0.1)
r0 = [r * a for r in ratios]
mag_w1 = [0.2, 1.0, 1.8]
mag_w2 = [0.1, 0.9, 1.7]
chi = [math.pi/3, math.pi/4, math.pi/5]
chi_show = [3, 4, 5]


#=======================================================#
# QBER parameters
#=======================================================#
    #=====================#
    # n_s   : average numher of photon
    # n_D   : dark-current-equivalent averrage photon number
    # eta   : APD (single-photon avalanche photodiode) detection efficiency
    # n_B   : background photons per polarization
    # n_N   : the average number of noise photon reaching each detector
    # gamma : the fraction of transmmited photon
    #======================#



def qber_loss(eta, n_s, n_N, gamma):
    term1 = np.exp(-eta * (n_s * gamma + 3 * n_N))
    term2 = 1 - np.exp(-eta * n_N)
    return term1 * term2


def to_decimal_string(x, precision=120):
    if x == 0:
        return "0." + "0" * precision
    return format(x, f'.{precision}f').rstrip('0').rstrip('.')

# 透過率 𝛾 が高いほど、エラー確率（QBER）は低くなる。
def main():
    n_s = 10e8
    n_D = math.pow(10, -6)
    eta = 0.5  
    n_B = math.pow(10, -3)
    n_N = n_B/2 + n_D
    gamma = 2.88503e-66
    # gamma = 6.068056215528627e-26
    prob_error = qber_loss(eta, n_s, n_N, gamma)
    print(f'QBER: {prob_error}')


if __name__ == "__main__":
    main()