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
#=======================================================#

#=======================================================#
# Beam waist function
#=======================================================#
def beam_waist(h_s, t):
    r_t = satellite_ground_distance(h_s, t)
    theta_d = 10e-6  # divergence angle(mrad)
    waist = r_t * theta_d
    return waist

#=======================================================#
# The distance between satellite and ground station
# h_s : Satellite's altitude
# t: time
#=======================================================#
def satellite_ground_distance(h_s, t): 
    d_o = D_E + h_s # orbital radius
    omega = math.sqrt(G * M_T / d_o**3)
    d_t = math.sqrt(D_E**2 + d_o**2 - 2 * D_E * d_o * math.cos(omega * t))
    return d_t
x

def main():
    # =======Definition of parameter =========== #
    h_s = 500e3 # (km)
    #====================================#
    # Need to be modify
    d_o = D_E + h_s
    omega = math.sqrt(G * M_T / d_o**3)
    T = 2 * math.pi / omega
    t = T * 0.03  # 周回の1/4周（90°移動）
    #====================================#
    waist = beam_waist(h_s, t)
    print(f'Beam waist: {waist}')

if __name__ == '__main__':
    main()