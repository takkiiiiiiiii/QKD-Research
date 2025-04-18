from scipy.special import lambertw, i0, i1
import math
import numpy as np

# These functions of the beam waist on the ground, W = Rθd > 4 m for satellites above 400 km.

# The beam transmissivity(defined as eta_b) consider beam spreading loss and pointing error. 

#=======================================================#
# circle beam transmissivity(eta_b) parameter
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
# Transmissivity nb(eta_b)
#=======================================================#
def transmissivity_etab(a, r, W):
    eta_0 = transmissivity_0(a, W)
    lambda_val = lambda_shape(a, W)
    R_val = r_scale(a, W)
    eta_b = eta_0 * np.exp(- (r / R_val) ** lambda_val)
    return eta_b
#=======================================================#


#=======================================================#
# Transmissivity n0(eta_0) : the transmittance for the
# centered beam
#=======================================================#
def transmissivity_0(a, W):
    ratio = a / W

    eta_0 = 1 - np.exp(-2 * ratio**2)

    return eta_0
#=======================================================#




#=======================================================#
# Scale function R
# Shape function lambda
#=======================================================#
def safe_i0(x):
    return np.where(x > 100, np.exp(x) / np.sqrt(2 * np.pi * x), i0(x))


def r_scale(a, W):
    exponent = 4 * (a ** 2) / (W ** 2)
    denominator = 1 - np.exp(-exponent) * i0(exponent)
    if denominator <= 0:
        raise ValueError("Denominator in logarithm is non-positive, check input values.")

    eta_0 = transmissivity_0(a, W)
    log_argument = (2 * eta_0) / denominator
    if log_argument <= 0:
        raise ValueError("Argument of logarithm is non-positive, check input values.")

    log_term = np.log(log_argument)
    lambda_val = lambda_shape(a, W)
    R = a * ((log_term) ** (-1 / lambda_val))

    return R

#=======================================================#


#=======================================================#
# Shape function lambda λ(ξ)
#=======================================================#
def safe_i1(x):
    return np.where(x > 100, np.exp(x) / np.sqrt(2 * np.pi * x), i1(x))  


def lambda_shape(a, W):
    exponent = 4 * (a ** 2) / (W ** 2)
    exp_term = np.exp(-exponent)
    I0_term = i0(exponent)
    I1_term = i1(exponent)
    denominator = 1 - exp_term * I0_term
    if denominator <= 0:
        raise ValueError("Denominator in lambda calculation is non-positive, check input values.")

    eta_0 = transmissivity_0(a, W)
    log_argument = (2 * eta_0) / denominator
    if log_argument <= 0:
        raise ValueError("Argument of logarithm is non-positive, check input values.")

    log_term = np.log(log_argument)

    lambda_val = (8 * (a ** 2) / (W ** 2)) * (exp_term * I1_term / denominator) * (1 / log_term)

    return lambda_val

#=======================================================#


def to_decimal_string(x, precision=70):
    if x == 0:
        return "0." + "0" * precision
    return format(x, f'.{precision}f').rstrip('0').rstrip('.')


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

def simulation_eta_b(h_s):
    # =======Definition of parameter =========== #
    # h_s = 500e3 # (km)
    #====================================#
    # Need to be modify
    d_o = D_E + h_s
    omega = math.sqrt(G * M_T / d_o**3)
    T = 2 * math.pi / omega
    t = T * 0.2  # 周回の1/4周（90°移動）
    #====================================#
    displacement = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    r = [a*d for d in displacement]
    waist = beam_waist(h_s, t)
    print(f'Beam waist: {waist}')
    W = [0.2*a, 1.0*a, 1.8*a]
    print("===============================")
    print(f'Aperture of radius (Receiver radis in meters): {a} m')
    # for i in range(len(W)):
    #         print(f"--- W[{i}] = {W[i]} ---")
    #         for displacement in r:
    #             eta_b = transmissivity_etab(a, displacement, W[i])
    #             print(f"r = {displacement}, eta_b = {eta_b}")
    for displacement in r:
        eta_b = transmissivity_etab(a, displacement, waist)
        print(f"r = {displacement}, eta_b = {eta_b}")
    print("===============================\n")
    
    
    print("Simulation Finish !!")


def main():
    h_s = 500e3
    simulation_eta_b(h_s)

if __name__ == '__main__':
    main()