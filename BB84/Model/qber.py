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
    # n_s   : average numher of photon from Alice
    # gamma : the fraction of transmmited photon
    # e_0   : the error rate of the background
    # Y_0   : the background rate which includes the detector dark count and other background contributions
    # e_dec : the probability that a photon hits the erroneous detector
    #======================#
n_s = 10^8
e_0 = 0.5
Y_0 = 10e-5
e_dec = 0.01

def qber_loss(gamma):
    ave_no_photon = n_s*gamma # average number of photons received by Bob
    qber = e_0 * Y_0 + e_dec*(1-np.exp(-ave_no_photon))
    return qber

def to_decimal_string(x, precision=120):
    if x == 0:
        return "0." + "0" * precision
    return format(x, f'.{precision}f').rstrip('0').rstrip('.')

# 透過率 𝛾 が高いほど、エラー確率（QBER）は低くなる。
def main():
    gamma = 
    # gamma = 6.068056215528627e-26
    prob_error = qber_loss(gamma)
    print(f'QBER: {prob_error}')


if __name__ == "__main__":
    main()