import numpy as np
import math


from atmospheric_transmissivity import transmissivity_etat
from circle_beam_transmissivity import satellite_ground_distance


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
r = 3

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
theta_d_rad =20e-6 

# #==================================================================#
# # v_wind: wind_speed
# #==================================================================#
v_wind = 21 
# #==================================================================#
# the maximum vertical altitude of atmosphere scaled from maximum 
# slant path h_slant,max over the atmosphere at minimum zenith angle
# atomospheric altitude
#==================================================================#
H_atm = 20e3

#==================================================================#
# waist : Beam waist radius at receiver (m)
#==================================================================#
# waist = beam_waist(h_s, H_a, theta_zen_rad, theta_d_rad)


#==================================================================#
# mu_x, mu_y, sigma_x, sigma_y
#==================================================================#
# mu_y = 0
# mu_x = 0
# sigma_x = 3e-6
# sigma_y = 3e-6


#==================================================================#
# I_a: Random intensity loss due to atmospheric turbulence
#==================================================================#
from scipy.stats import lognorm

def compute_intensity_loss(sigma_R_squared, size=1):
    """
    Compute random intensity loss due to atmospheric turbulence using a log-normal distribution.
    
    Parameters:
    - sigma_R_squared: Rytov variance, which characterizes the turbulence strength.
    - size: The number of random samples to generate.
    
    Returns:
    - Random intensity loss values as an array.
    """
    # Compute the parameters for the log-normal distribution
    shape_param = np.sqrt(sigma_R_squared)  # Shape parameter for log-normal distribution
    
    # Generate random intensity loss values using log-normal distribution
    I_a_random = lognorm.rvs(shape_param, loc=0, scale=1, size=size)
    
    return I_a_random


#==================================================================#
# Compute eta_p : beam misalignment and spreading loss
#==================================================================#
def transmissivity_etap(theta_zen_rad, lambda_, h_s, H_g, Cn2_profile, a):
    L = satellite_ground_distance(h_s, H_g, theta_zen_rad)

    w_L = compute_w_L(lambda_, theta_d_half_rad, L, H_atm, H_g, theta_zen_rad, Cn2_profile)

    w_Leq_squared = equivalent_beam_width_squared(a, w_L)
    nu = (math.sqrt(math.pi) * a) / (math.sqrt(2) * w_L)

    A0 = erf(nu)**2
    eta_p = A0 * np.exp(-(2*r**2)/(w_Leq_squared))
    return eta_p


#==================================================================#
# Compute eta : Combined transmittivity
#==================================================================#
def compute_eta(eta_t, sigma_R_squared, theta_zen_rad, lambda_, h_s, H_g, Cn2_profile, a, size=1):
    """
    Compute the combined transmittivity eta, considering atmospheric attenuation, 
    random intensity fluctuations, and pointing errors/divergence losses.
    
    Parameters:
    - eta_t: Fixed atmospheric attenuation factor.
    - sigma_R_squared: Rytov variance, characterizing turbulence strength.
    - theta_zen_rad: Zenith angle in radians.
    - lambda_: Wavelength of the signal.
    - h_s: Height of the satellite.
    - H_g: Height of the ground station.
    - Cn2_profile: Profile of the refractive index structure constant.
    - a: Aperture radius.
    - size: Number of random samples to generate for intensity loss.
    
    Returns:
    - eta: The combined transmittivity (including random intensity loss and pointing/divergence loss).
    """
    # Step 1: Compute random intensity loss due to atmospheric turbulence (I_a)
    I_a_random = compute_intensity_loss(sigma_R_squared, size)
    
    # Step 2: Compute eta_p (beam misalignment and spreading loss)
    eta_p = transmissivity_etap(theta_zen_rad, lambda_, h_s, H_g, Cn2_profile, a)
    
    # Step 3: Calculate the final transmittivity eta
    eta = eta_t * I_a_random * eta_p
    
    return eta

# Example usage
eta_t = 0.9  # Fixed atmospheric attenuation (example)
sigma_R_squared = 0.5  # Example Rytov variance (weak turbulence)
theta_zen_rad = np.radians(30)  # Zenith angle in radians
lambda_ = 1550e-9  # Wavelength (example)
h_s = 500e3  # Satellite altitude (example)
H_g = 0  # Ground station altitude (example)
Cn2_profile = None  # Placeholder for refractive index profile
a = 0.75  # Aperture radius (example)

# Compute eta for 10 random samples
eta_values = compute_eta(eta_t, sigma_R_squared, theta_zen_rad, lambda_, h_s, H_g, Cn2_profile, a, size=10)

print("Computed Transmittivity (eta) Values:", eta_values)
