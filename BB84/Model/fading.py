import math
import numpy as np
from scipy.integrate import quad
from circle_beam_transmissivity import transmissivity_0, beam_waist
from atmospheric_transmissivity import transmissivity_etat
from circle_beam_transmissivity import transmissivity_etab


#=======================================================#
#                 Fading Parameters                     #
#=======================================================#
#=====================#
# a : Aperture of radius (Receiver radis in meters)
#=====================#
a = 0.75

#=====================#
# r : Radial jitter distance
#=====================#
r = 5

#=====================#
# len_wave : Optical wavelength (μm)
#=====================#
len_wave = 0.85  # (μm)

#=====================#
# sigma_R : Rytov variance (乱流の強さの指標)
#=====================#
sigma_R = 1.0  # 中程度の乱流

#=====================#
# h_s : Altitude between LEO satellite and ground station (m)
#=====================#
h_s = 500e3  # 500 km

#=====================#
# H_a : Upper end of atmosphere (km)
#=====================#
H_a = 0.01  # 10 m (大気の終端高度)

#=====================#
# tau_zen : Transmission efficiency at zenith
#=====================#
tau_zen = 0.85  # 天頂方向での大気透過率

#=====================#
# theta_zen_rad : Zenith angle (radian)
#=====================#
theta_zen_rad = math.radians(30)  # ラジアン変換

#=====================#
# theta_d_rad : Optical beam divergence angle (radian)
#=====================#
theta_d_rad = 10e-6  # ビームの発散角

#=====================#
# waist : Beam waist radius at receiver (m)
#=====================#
waist = beam_waist(h_s, H_a, theta_zen_rad, theta_d_rad)


#==================================================================#
# Beam jitter's means (mu_x, mu_y) and variances (sigma_x, sigma_y)
# weak -> ズレが小さい (精密な追尾・安定な環境)
# theta : Optical beam divergence half-angle at exp(-2)
#==================================================================#
theta_d_half_rad = theta_d_rad / 2

jitter_params = {
    "weak": {
        "mu_x": 0*h_s,
        "mu_y": 0*h_s,
        "sigma_x": theta_d_half_rad/5 * h_s,
        "sigma_y": theta_d_half_rad/5 * h_s
    },
    "moderate": {
        "mu_x": theta_d_half_rad/5 * h_s,
        "mu_y": theta_d_half_rad/3 * h_s,
        "sigma_x": theta_d_half_rad/2 * h_s,
        "sigma_y": theta_d_half_rad/3 * h_s
    },
    "strong": {
        "mu_x": theta_d_half_rad/5 * h_s,
        "mu_y": theta_d_half_rad/3 * h_s,
        "sigma_x": theta_d_half_rad/1.5 * h_s,
        "sigma_y": theta_d_half_rad/2 * h_s
    }
}

# Level of pointing error
condition = "moderate"

params = jitter_params[condition]
mu_x = params["mu_x"]
mu_y = params["mu_y"]
sigma_x = params["sigma_x"]
sigma_y = params["sigma_y"]



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

# Complementary error function for 'fading_loss()'
def erfc(x):
    integral, _ = quad(lambda t: np.exp(-t**2), x, np.inf)
    return (2 / np.sqrt(np.pi)) * integral


def fading_loss(gamma, mu_x, mu_y, sigma_x, sigma_y):
    eta_t = transmissivity_etat(tau_zen, theta_zen_rad)
    eta_b = transmissivity_etab(a, r, waist)
    sigma_mod = approximate_jitter_variance(mu_x, mu_y, sigma_x, sigma_y)
    varphi_mod = sigma_to_variance(sigma_mod)
    gamma= eta_t * eta_b
    A_mod = mod_jitter()
    mu = sigma_R**2/2 * (1+2*varphi_mod**2)
    term1 = (varphi_mod**2) / (2 * (A_mod * eta_t)**(varphi_mod**2))
    term2 = gamma**(varphi_mod**2**-1)
    term3 = erfc((np.log((gamma / (A_mod * eta_t))) + mu) / (np.sqrt(2) * sigma_R))
    term4 = np.exp((sigma_R**2) / 2 * varphi_mod**2 * (1 + varphi_mod**2))
    
    eta_f = term1 * term2 * term3 * term4
    return eta_f


def main():
    eta_t = transmissivity_etat(tau_zen, theta_zen_rad)
    eta_b = transmissivity_etab(a, r, waist)
    gamma= eta_t * eta_b
    fading_val = fading_loss(gamma)
    print(f'Fadeing loss: {fading_val}')


if __name__ == '__main__':
    main()