import math
import numpy as np
from scipy.integrate import quad
from scipy.special import erfc
from scipy.special import erf
from circle_beam_transmissivity import transmissivity_0, beam_waist
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
theta_zen_rad = math.radians(10)

# #==================================================================#
# # theta_d_rad : Optical beam divergence angle (rad)
# #==================================================================#
theta_d_rad = 10e-6 

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

theta_d_half_rad = theta_d_rad / 2
# def get_beam_jitter_params(condition, theta_d_rad, L):
#     theta_d_half_rad = theta_d_rad / 2
#     # theta_d_half_rad = 16.5e-6

#     jitter_params = {
#         "weak": {
#             "mu_x": 0 * L,
#             "mu_y": 0 * L,
#             "sigma_x": theta_d_half_rad / 5 * L,
#             "sigma_y": theta_d_half_rad / 5 * L
#         },
#         "moderate": {
#             "mu_x": theta_d_half_rad / 5 * L,
#             "mu_y": theta_d_half_rad / 3 * L,
#             "sigma_x": theta_d_half_rad / 2 * L,
#             "sigma_y": theta_d_half_rad / 3 * L
#         },
#         "strong": {
#             "mu_x": theta_d_half_rad / 5 * L,
#             "mu_y": theta_d_half_rad / 3 * L,
#             "sigma_x": theta_d_half_rad / 1.5 * L,
#             "sigma_y": theta_d_half_rad / 2 * L
#         }
#     }

#     if condition not in jitter_params:
#         raise ValueError("Invalid condition. Choose from 'weak', 'moderate', or 'strong'.")

#     return jitter_params[condition]

def Cn_squared(h):
    return 1e-14

# Beam footprint radius at receiver including turbulence
def compute_w_L(lambda_, theta_d_half_rad, L, H_atm, H_OGS, theta_zen_rad):
    k = 2 * math.pi / lambda_

    w_0 = lambda_ / (math.pi * theta_d_half_rad)

    W = w_0 * math.sqrt(1 + (2 * L) / (k * w_0))


    def integrand(h):
        return Cn_squared(h) * ((h - H_OGS) / (H_atm - H_OGS))**(5/3)

    integral_result, _ = quad(integrand, H_OGS, H_atm)

    T = 4.35 * ((2 * L) / (k * W**2))**(5/6) * \
        k**(7/6) * (H_atm - H_OGS)**(5/6) * \
        (1 / math.cos(theta_zen_rad))**(11/6) * integral_result

    # Step 5: compute final beam radius at receiver
    w_L = W * math.sqrt(1 + T)
    return w_L


# calculate modified beam-jitter variance approximation
def approximate_jitter_variance(mu_x, mu_y, sigma_x, sigma_y):
    numerator = (
        3 * mu_x**2 * sigma_x**4 +
        3 * mu_y**2 * sigma_y**4 +
        sigma_x**6 +
        sigma_y**6
    )
    sigma_mod_value = (numerator / 2) ** (1/3)
    return sigma_mod_value

# calculate the ratios between the equivalent beam-width and (modified) beam-jitter variances
def sigma_to_variance(sigma, waist):
    variance = waist/2*sigma
    return variance

# calculate the modified fracton of collected power over the receiving aparture when there is no pointing error
def mod_jitter(mu_x, mu_y, sigma_x, sigma_y, waist):
    A_0 = transmissivity_0(a, waist)
    varphi_x = sigma_to_variance(sigma_x, waist)
    varphi_y = sigma_to_variance(sigma_y, waist)
    sigma_mod = approximate_jitter_variance(mu_x, mu_y, sigma_x, sigma_y)
    varphi_mod = sigma_to_variance(sigma_mod, waist)
    term1 = 1 / (varphi_mod ** 2)
    term2 = 1 / (2 * varphi_x ** 2)
    term3 = 1 / (2 * varphi_y ** 2)
    term4 = mu_x**2 / (2 * sigma_x ** 2 * varphi_x ** 2)
    term5 = mu_y**2 / (2 * sigma_y ** 2 * varphi_y ** 2)
    exponent = term1 - term2 - term3 - term4 - term5
    A_mod = A_0 * np.exp(exponent)
    return A_mod


# Compute Rytov variance σ_R^2 for atmospheric turbulence.
def rytov_variance(len_wave, theta_zen_rad, H_OGS, H_atm, Cn2_profile):
    k = 2 * np.pi / len_wave
    sec_zenith = 1 / np.cos(theta_zen_rad)

    def integrand(h):
        return Cn2_profile(h) * (h - H_OGS)**(5/6)

    integral, _ = quad(integrand, H_OGS, H_atm, limit=100, epsabs=1e-9, epsrel=1e-9)

    sigma_R_squared = 2.25 * (k)**(7/6) * sec_zenith**(11/6) * integral

    return sigma_R_squared


def cn2_profile(h, v_wind=21, Cn2_0=1e-17):
    term1 = 0.00594 * (v_wind / 27)**2 * (1e-5 * h)**10 * np.exp(-h / 1000)
    term2 = 2.7e-16 * np.exp(-h / 1500)
    term3 = Cn2_0 * np.exp(-h / 100)
    return term1 + term2 + term3


# Calculate sigma_mod
def compute_sigma_mod(mu_x, mu_y, sigma_x, sigma_y):
    numerator = (
        3 * mu_x**2 * sigma_x**4 +
        3 * mu_y**2 * sigma_y**4 +
        sigma_x**6 +
        sigma_y**6
    )
    sigma_mod = (numerator / 2) ** (1/3)
    return sigma_mod

# Equivalent Beam Width
def equivalent_beam_width_squared(a, w_L):
    nu = (math.sqrt(math.pi) * a) / (math.sqrt(2) * w_L)
    numerator = math.sqrt(math.pi) * erf(nu)
    denominator = 2 * nu * math.exp(-nu**2)
    w_Leq_squared = w_L**2 * (numerator / denominator)
    return w_Leq_squared

# Calculate varphi_mod
def varphi_mod(w_Leq_squared, sigma_mod):
    w_Leq = math.sqrt(w_Leq_squared)
    return w_Leq / (2 * sigma_mod)

# Calculate the fading loss value
def fading_loss(gamma, mu_x, mu_y, sigma_x, sigma_y, theta_zen_rad, H_atm, waist):
    eta_t = transmissivity_etat(tau_zen, theta_zen_rad)
    sigma_R_squared = rytov_variance(len_wave, theta_zen_rad, H_ogs, H_atm, cn2_profile)
    varphi_mod = 4.3292
    A_mod = mod_jitter(mu_x, mu_y, sigma_x, sigma_y, waist)
    mu = sigma_R_squared/2 * (1+2*varphi_mod**2)
    term1 = (varphi_mod**2) / (2 * (A_mod * eta_t)**(varphi_mod**2))
    term2 = gamma ** (varphi_mod**2 - 1)
    term3 = erfc((np.log((gamma / (A_mod * eta_t))) + mu) / (np.sqrt(2) * math.sqrt(sigma_R_squared)))
    term4 = np.exp(((sigma_R_squared/2) * varphi_mod**2 * (1 + varphi_mod**2)))
    eta_f = term1 * term2 * term3 * term4
    return eta_f

def to_decimal_string(x, precision=100):
    if x == 0:
        return "0." + "0" * precision
    return format(x, f'.{precision}f').rstrip('0').rstrip('.')

def main():
    # beam propagation distance
    L = satellite_ground_distance(h_s, H_g, theta_zen_rad)
    # beam jitter param
    mu_y = 0
    mu_x = 0
    angle_sigma_x = 3e-6
    angle_sigma_y = 3e-6
    sigma_x = angle_sigma_x * L
    sigma_y = angle_sigma_y * L

    w_L = compute_w_L(lambda_, theta_d_half_rad, L, H_atm, H_g, theta_zen_rad)

    sigma_mod = compute_sigma_mod(mu_x, mu_y, sigma_x, sigma_y)

    w_Leq_squared = equivalent_beam_width_squared(a, w_L)

    varphi = varphi_mod(w_Leq_squared, sigma_mod)

    # Output result
    print(f"sigma_mod: {sigma_mod:.3e} [m]")
    print(f"w_Leq_squared: {w_Leq_squared:.3e} [m^2]")
    print(f"varphi_mod: {varphi:.3f}")


if __name__ == '__main__':
    main()