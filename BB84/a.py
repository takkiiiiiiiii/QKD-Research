import math
import numpy as np
from scipy.integrate import quad
from scipy.special import erfc
from scipy.special import erf
from scipy.stats import lognorm
import matplotlib.pyplot as plt
from scipy.integrate import quad


#=======================================================#
#                 Fading Parameters                     #
#=======================================================#
#==================================================================#
# a : Aperture of radius (Receiver radis in meters) (m)
#==================================================================#
# a = 0.07920
a = 0.75

# r = 5
#==================================================================#
# n_s : average number of photon
#==================================================================#
n_s = 0.8

#==================================================================#
# len_wave : Optical wavelength (μm)
#==================================================================#
lambda_ = 0.85e-6

#==================================================================#
# altitude ground station
#==================================================================#
H_g = 10 # (m)

#==================================================================#
# h_s : Altitude between LEO satellite and ground station (m)
#==================================================================#
h_s = 550e3  # 500 km

#==================================================================#
# H_a : Upper end of atmosphere (km)
#==================================================================#
H_atm = 200000

#==================================================================#
# theta_d_rad : Optical beam divergence angle (rad)
#==================================================================#
theta_d_rad = 10e-6 

#==================================================================#
# theta_d_half_rad : Optical beam half-divergence angle (rad)
#==================================================================#
theta_d_half_rad = theta_d_rad /2

#==================================================================#
# v_wind: wind_speed
#==================================================================#
v_wind = 21 

#==================================================================#
# mu_x, mu_y: Mean values of pointing error in x and y directions (m)
#==================================================================#
mu_x = 0
mu_y = 0

#==================================================================#
# angle_sigma_x, angle_sigma_y: Beam jitter standard deviations of the Gaussian-distibution jitters (rad)
#==================================================================#
angle_sigma_x = 3e-6
angle_sigma_y = 3e-6

#=======================================================#
# QBER parameters
#=======================================================#
    #=====================#
    # n_s   : average numher of photon from Alice
    # e_0   : the error rate of the background
    # Y_0   : the background rate which includes the detector dark count and other background contributions
    # P_pa  : After-pulsing probability
    # e_pol : Probability of the polarisation errors
    #======================#i
e_0 = 0.5
Y_0 = 1e-4
P_AP = 0.02
e_pol = 0.033

theta_zen_rad = math.radians(20)

#==================================================================#
# Beam footprint radius at receiver including turbulence
#==================================================================#
def compute_w_L(lambda_, theta_d_half_rad, L, H_atm, H_OGS, theta_zen_rad, Cn2_profile):
    k = 2 * math.pi /lambda_
    w0 = lambda_ * (math.pi * theta_d_half_rad)**(-1)
    w0_squared = w0**2
    W = w0 * math.sqrt(1+(2*L)/k*w0_squared)

    def integrand(h):
        return Cn2_profile(h) * ((h - H_OGS) / (H_atm - H_OGS))**(5/3)
    
    integral_result, _ = quad(integrand, H_OGS, H_atm)

    T = 4.35 * ((2 * L) / (k * W**2))**(5/6) * \
        k**(7/6) * (H_atm - H_OGS)**(5/6) * \
        (1 / math.cos(theta_zen_rad))**(11/6) * integral_result

    w_L = W * math.sqrt(1+T)
    return w_L


def Cn2_profile(h, v_wind=21, Cn2_0=1e-13):
    term1 = 0.00594 * (v_wind / 27)**2 * (1e-5 * h)**10 * np.exp(-h / 1000)
    term2 = 2.7e-16 * np.exp(-h / 1500)
    term3 = Cn2_0 * np.exp(-h / 100)
    return term1 + term2 + term3

#==================================================================#
# Compute beam propagation disteance
#==================================================================#
def satellite_ground_distance(h_s, H_g, theta_zen_rad):
    return (h_s - H_g) / math.cos(theta_zen_rad)

def main():
    LoS = satellite_ground_distance(h_s, H_g, theta_zen_rad)
    w_L = compute_w_L(lambda_, theta_d_half_rad, LoS, H_atm, H_g, 
    theta_zen_rad, Cn2_profile)
    print(f'w_L :{w_L:.3e} [m]')
    
if __name__ == "__main__":
    main()
