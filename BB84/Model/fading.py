import math
import numpy as np
from scipy.integrate import quad
from scipy.special import erfc
from circle_beam_transmissivity import transmissivity_0, beam_waist
from atmospheric_transmissivity import transmissivity_etat
from circle_beam_transmissivity import transmissivity_etab


#=======================================================#
#                 Fading Parameters                     #
#=======================================================#
#==================================================================#
# a : Aperture of radius (Receiver radis in meters) (m)
#==================================================================#
a = 0.75

#==================================================================#
# r : Radial jitter distance (m)
#==================================================================#
r = 5

#==================================================================#
# len_wave : Optical wavelength (μm)
#==================================================================#
len_wave = 0.85

#==================================================================#
# sigma_R : Rytov variance (乱流の強さの指標)
#==================================================================#
sigma_R_squared = 1.0

#==================================================================#
# Upper atmospheric height limit (m)
#==================================================================#
H_atm = 20000 # 20km

#==================================================================#
# h_s : Altitude between LEO satellite and ground station (m)
#==================================================================#
h_s = 500e3  # 500 km

#==================================================================#
# H_a : Upper end of atmosphere (km)
#==================================================================#
H_a = 0.01  # 10 m (大気の終端高度)

#==================================================================#
# tau_zen : Transmission efficiency at zenith
#==================================================================#
tau_zen = 0.85  # 天頂方向での大気透過率

#==================================================================#
# theta_zen_rad : Zenith angle (rad)
#==================================================================#
theta_zen_rad = math.radians(30)

#==================================================================#
# theta_d_rad : Optical beam divergence angle (rad)
#==================================================================#
theta_d_rad = 10e-6 

#==================================================================#
# waist : Beam waist radius at receiver (m)
#==================================================================#
waist = beam_waist(h_s, H_a, theta_zen_rad, theta_d_rad)


def get_beam_jitter_params(condition, theta_d_rad):
    theta_d_half_rad = theta_d_rad / 2
    # theta_d_half_rad = 16.5e-6

    jitter_params = {
        "weak": {
            "mu_x": 0 * h_s,
            "mu_y": 0 * h_s,
            "sigma_x": theta_d_half_rad / 5 * h_s,
            "sigma_y": theta_d_half_rad / 5 * h_s
        },
        "moderate": {
            "mu_x": theta_d_half_rad / 5 * h_s,
            "mu_y": theta_d_half_rad / 3 * h_s,
            "sigma_x": theta_d_half_rad / 2 * h_s,
            "sigma_y": theta_d_half_rad / 3 * h_s
        },
        "strong": {
            "mu_x": theta_d_half_rad / 5 * h_s,
            "mu_y": theta_d_half_rad / 3 * h_s,
            "sigma_x": theta_d_half_rad / 1.5 * h_s,
            "sigma_y": theta_d_half_rad / 2 * h_s
        }
    }

    if condition not in jitter_params:
        raise ValueError("Invalid condition. Choose from 'weak', 'moderate', or 'strong'.")

    return jitter_params[condition]


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
def sigma_to_variance(sigma):
    variance = waist/2*sigma
    return variance

# calculate the modified fracton of collected power over the receiving aparture when there is no pointing error
def mod_jitter(mu_x, mu_y, sigma_x, sigma_y):
    A_0 = transmissivity_0(a, waist)
    varphi_x = sigma_to_variance(sigma_x)
    varphi_y = sigma_to_variance(sigma_y)
    sigma_mod = approximate_jitter_variance(mu_x, mu_y, sigma_x, sigma_y)
    varphi_mod = sigma_to_variance(sigma_mod)
    term1 = 1 / (varphi_mod ** 2)
    term2 = 1 / (2 * varphi_x ** 2)
    term3 = 1 / (2 * varphi_y ** 2)
    term4 = mu_x**2 / (2 * sigma_x ** 2 * varphi_x ** 2)
    term5 = mu_y**2 / (2 * sigma_y ** 2 * varphi_y ** 2)
    exponent = term1 - term2 - term3 - term4 - term5
    A_mod = A_0 * np.exp(exponent)
    return A_mod


# Compute Rytov variance σ_R^2 for atmospheric turbulence.
def rytov_variance(lambda_m, theta_zen_rad, H_OGS, H_atm, Cn2_profile):
    k = 2 * np.pi / lambda_m
    sec_zenith = 1 / np.cos(theta_zen_rad)

    def integrand(h):
        return Cn2_profile(h) * (h - H_OGS)**(5/6)

    integral, _ = quad(integrand, H_OGS, H_atm, limit=100, epsabs=1e-9, epsrel=1e-9)

    sigma_R_squared = 2.25 * (k)**(7/6) * sec_zenith**(11/6) * integral

    return sigma_R_squared

def simple_cn2_profile(h):
    """ A simple model for Cn^2(h) [m^-2/3] """
    return 1e-14 * np.exp(-h / 1000) 

# Calculate the fading loss value
def fading_loss(gamma, mu_x, mu_y, sigma_x, sigma_y):
    eta_t = transmissivity_etat(tau_zen, theta_zen_rad)
    # eta_b = transmissivity_etab(a, r, waist)
    # sigma_mod = approximate_jitter_variance(mu_x, mu_y, sigma_x, sigma_y)
    # varphi_mod = sigma_to_variance(sigma_mod)
    varphi_mod = 4.3292
    A_mod = mod_jitter(mu_x, mu_y, sigma_x, sigma_y)
    mu = sigma_R_squared/2 * (1+2*varphi_mod**2)
    term1 = (varphi_mod**2) / (2 * (A_mod * eta_t)**(varphi_mod**2))
    # print(f'term1: {term1}')
    term2 = gamma ** (varphi_mod**2 - 1)
    # term2 = gamma ** (1 / (varphi_mod**2))
    # print(f'term2: {term2}') # 0.7630630105810166
    term3 = erfc((np.log((gamma / (A_mod * eta_t))) + mu) / (np.sqrt(2) * math.sqrt(sigma_R_squared)))
    # print(f'(np.log((gamma / (A_mod * eta_t))) + mu) / (np.sqrt(2) * math.sqrt(sigma_R_squared)): {(np.log((gamma / (A_mod * eta_t))) + mu) / (np.sqrt(2) * math.sqrt(sigma_R_squared))}') #
    # print(f'np.log((gamma / (A_mod * eta_t))) + mu: {np.log((gamma / (A_mod * eta_t))) + mu}')
    # print(f'varphi_mod: {varphi_mod}')
    term4 = np.exp(((sigma_R_squared/2) * varphi_mod**2 * (1 + varphi_mod**2)))
    # print(f'((sigma_R_squared) * varphi_mod**2 * (1 + varphi_mod**2) / 2): {((sigma_R_squared) * varphi_mod**2 * (1 + varphi_mod**2) / 2)}')
    # print(f'term4: {term4}')
    
    eta_f = term1 * term2 * term3 * term4
    return eta_f

def to_decimal_string(x, precision=100):
    if x == 0:
        return "0." + "0" * precision
    return format(x, f'.{precision}f').rstrip('0').rstrip('.')

def main():
    # conditon: strong, moderate or weak
    params = get_beam_jitter_params(condition="strong", theta_d_rad=theta_d_rad)
    mu_x = params["mu_x"]
    mu_y = params["mu_y"]
    sigma_x = params["sigma_x"]
    sigma_y = params["sigma_y"]
    eta_t = transmissivity_etat(tau_zen, theta_zen_rad)
    eta_b = transmissivity_etab(a, r, waist)
    gamma= eta_t * eta_b

    fading_val = fading_loss(gamma, mu_x, mu_y, sigma_x, sigma_y)
    print(f'Fading loss: {to_decimal_string(fading_val)}')


if __name__ == '__main__':
    main()