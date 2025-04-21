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
# Channel loss parameter
#=======================================================#
    #=====================#
    # n_s   : average numher of photon
    # n_D   : dark-current-equivalent averrage photon number
    # eta   : APD (single-photon avalanche photodiode) detection efficiency
    # n_B   : background photons per polarization
    # n_N   : the average number of noise photon reaching each detector
    # gamma : the fraction of transmmited photon
    #======================#
n_s = 20
n_D = math.pow(10, -6)
eta = 0.5  
n_B = math.pow(10, -3)
n_N = n_B/2 + n_D


def qber_loss(eta, n_s, n_N, gamma):
    term1 = np.exp(-eta * (n_s * gamma + 3 * n_N))
    term2 = 1 - np.exp(-eta * n_N)
    return term1 * term2


def main():
    gamma = 4.00423e-02
    # gamma = 6.068056215528627e-26
    prob_error = qber_loss(gamma)
    print(prob_error)


if __name__ == "__main__":
    main()