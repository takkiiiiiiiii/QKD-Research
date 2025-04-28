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
    # e_0   : the error rate of the background
    # Y_0   : the background rate which includes the detector dark count and other background contributions
    # P_pa  : After-pulsing probability
    # e_pol : Probability of the polarisation errors
    #======================#
n_s = 0.1
e_0 = 0.5
Y_0 = 10e-4
P_pa = 0.02
e_pol = 0.033

def qber_loss(gamma):
    denominator =  e_0 * Y_0 + (e_pol*e_0*P_pa) * (1-np.exp(-n_s*gamma))
    numerator = Y_0 + (1-np.exp(-n_s*gamma)) * (1+P_pa)

    qber = denominator/numerator
    return qber

def to_decimal_string(x, precision=120):
    if x == 0:
        return "0." + "0" * precision
    return format(x, f'.{precision}f').rstrip('0').rstrip('.')

def main():
    gamma = 0.04
    prob_error = qber_loss(gamma) * 100
    print(f'QBER: {prob_error} %')

if __name__ == "__main__":
    main()