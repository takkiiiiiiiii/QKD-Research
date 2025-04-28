import math
import numpy as np


#=======================================================#
# Beam waist parameter
#=======================================================#
    #=====================#
    # D_r    : Deceiver diameter in meters
    # a      : Aperture of radius (Receiver radis in meters)
    #======================#
# D_r = 1.5 #(a = 0.75m)
a = 0.75
# Constant
G = 6.67430e-11         # Gravitational constant
M_T = 5.972e24          # Earth's mass
D_E = 6378e3             # Earth's radius (km)
H_g = 0.01
#=======================================================#

#=======================================================#
# Beam waist function
#=======================================================#
def beam_waist(h_s, H_a, theta_zen_rad):
    L_a = satellite_ground_distance(h_s, H_a, theta_zen_rad)
    theta_d = 10e-6  # divergence angle(mrad)
    waist = L_a * theta_d
    return waist


#=======================================================#
# The distance between satellite and ground station
# h_s       : Satellite's altitude
# theta_zen : Zenith angle
#=======================================================#
def satellite_ground_distance(h_s, H_g, theta_zen_rad):
    return (h_s - H_g) / math.cos(theta_zen_rad)

def main():
    # =======Definition of parameter =========== #
    h_s = 500e3 # (km)
    theta_zen_rad = math.radians(7.5)
    #====================================#
    L_a = satellite_ground_distance(h_s, H_g, theta_zen_rad)
    print(f'LoS distance: {L_a/1e3:.1f} km')
    waist = beam_waist(h_s, theta_zen_rad)
    print(f'Beam waist: {waist}')

if __name__ == '__main__':
    main()