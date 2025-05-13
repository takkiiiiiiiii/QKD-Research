import math
import numpy as np
from scipy.integrate import quad
from scipy.special import erfc
from scipy.special import erf
from circle_beam_transmissivity import transmissivity_0, beam_waist
from atmospheric_transmissivity import transmissivity_etat
from circle_beam_transmissivity import satellite_ground_distance
from fading import compute_w_L, equivalent_beam_width_squared

#=======================================================#
#                 Fading Parameters                     #
#=======================================================#
#==================================================================#
# a : Aperture of radius (Receiver radis in meters) (m)
#==================================================================#
a = 0.75

# #==================================================================#
# # r : Radial jitter distance (m)
# #==================================================================#
r = 5

# #==================================================================#
# # len_wave : Optical wavelength (μm)
# #==================================================================#
lambda_ = 0.85e-6

# #==================================================================#
# # altitude ground station
# #==================================================================#
H_g = 10 # (m)

# #==================================================================#
# # h_s : Altitude between LEO satellite and ground station (m)
# #==================================================================#
h_s = 550e3  # 500 km

# #==================================================================#
# # H_a : Upper end of atmosphere (km)
# #==================================================================#
H_a = 0.01  # 10 m (大気の終端高度)

# #==================================================================#
# # tau_zen : Transmission efficiency at zenith
# #==================================================================#
tau_zen = 0.85  # 天頂方向での大気透過率

# #==================================================================#
# # theta_zen_rad : Zenith angle (rad)
# #==================================================================#
theta_zen_rad = math.radians(40)

# #==================================================================#
# # theta_d_rad : Optical beam divergence angle (rad)
# #==================================================================#
theta_d_rad = 10e-6 
theta_d_half_rad = theta_d_rad/2

# #==================================================================#
# # v_wind: wind_speed
# #==================================================================#
v_wind = 21 
# #==================================================================#
# the maximum vertical altitude of atmosphere scaled from maximum 
# slant path h_slant,max over the atmosphere at minimum zenith angle
# atomospheric altitude
#==================================================================#
H_atm = 20000

def main():
    # #==================================================================#
    # # beam propagation distance (LoS)
    # #==================================================================#
    L = satellite_ground_distance(h_s, H_g, theta_zen_rad)

    # #==================================================================#
    # # w_L: Beam footprint radius
    # #==================================================================#
    w_L = compute_w_L(lambda_, theta_d_half_rad, L, H_atm, H_g, theta_zen_rad)

    w_Leq_squared = equivalent_beam_width_squared(a, w_L)
    nu = (math.sqrt(math.pi) * a) / (math.sqrt(2) * w_L)

    A0 = erf(nu)**2
    eta_p = A0 * np.exp(-(2*r**2)/(w_Leq_squared))
    print(eta_p)

if __name__ == "__main__":
    main()