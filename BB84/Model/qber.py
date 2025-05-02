from scipy.special import lambertw, i0, i1
import math
import numpy as np
from circle_beam_transmissivity import transmissivity_etab, beam_waist
from atmospheric_transmissivity import transmissivity_etat

a = 0.75                    
h_s = 500e3       
H_g = 10
theta_d_rad = 10e-6

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
e_0 = 0.5
Y_0 = 1e-4
P_AP = 0.02
e_pol = 0.033

def qber_loss(gamma, n_s):
    denominator =  e_0 * (Y_0*(1+P_AP)) + (e_pol+e_0*P_AP) * (1-np.exp(-n_s*gamma))
    numerator = (Y_0*(1+P_AP)) + (1-np.exp(-n_s*gamma)) * (1+P_AP)

    qber = denominator/numerator
    return qber

def to_decimal_string(x, precision=120):
    if x == 0:
        return "0." + "0" * precision
    return format(x, f'.{precision}f').rstrip('0').rstrip('.')

def main():
    n_s = 0.1
    r = 4
    tau_zen = 0.91
    theta_zen_rad = math.radians(60)
    waist = beam_waist(h_s, H_g, theta_zen_rad, theta_d_rad)
    eta_t = transmissivity_etat(tau_zen, theta_zen_rad)
    eta_b = transmissivity_etab(a, r, waist)
    print(f'eta_b {eta_b}')
    gamma = eta_b * eta_t
    prob_error = qber_loss(gamma, n_s) * 100
    print(f'QBER: {prob_error} %')

if __name__ == "__main__":
    main()