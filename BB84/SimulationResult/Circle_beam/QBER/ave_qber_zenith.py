import math
import numpy as np
from scipy.integrate import quad
from scipy.special import erfc
import matplotlib.pyplot as plt
from scipy.integrate import quad
import os, sys

# モジュール読み込み
simulation_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../', 'Model'))
sys.path.append(simulation_path)

from circle_beam_transmissivity import transmissivity_0, satellite_ground_distance, beam_waist
from atmospheric_transmissivity import transmissivity_etat
from qber import qber_loss


#=======================================================#
#                 Fading Parameters                     #
#=======================================================#
#==================================================================#
# a : Aperture of radius (Receiver radis in meters) (m)
#==================================================================#
a = 0.75

#==================================================================#
# varphi_mod : Beam width to jitter variance ratio 
#==================================================================#
varphi_mod = 4.3292

#==================================================================#
# n_s : average number of photon
#==================================================================#
n_s = 0.8

#==================================================================#
# len_wave : Optical wavelength (μm)
#==================================================================#
len_wave = 0.85e-6

#==================================================================#
# altitude ground station
#==================================================================#
H_ogs = 10 # (m)

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
# tau_zen = 0.91  # 天頂方向での大気透過率

#==================================================================#
# theta_zen_rad : Zenith angle (rad)
#==================================================================#
# theta_zen_rad = math.radians(10)

#==================================================================#
# theta_d_rad : Optical beam divergence angle (rad)
#==================================================================#
theta_d_rad = 10e-6 

#==================================================================#
# v_wind: wind_speed
#==================================================================#
v_wind = 21 
#==================================================================#
# the maximum vertical altitude of atmosphere scaled from maximum 
# slant path h_slant,max over the atmosphere at minimum zenith angle
# atomospheric altitude
#==================================================================#
# H_atm = 20000*math.cos(theta_zen_rad) 

#==================================================================#
# waist : Beam waist radius at receiver (m)
#==================================================================#
# waist = beam_waist(h_s, H_a, theta_zen_rad, theta_d_rad)




def get_beam_jitter_params(condition, theta_d_rad):
    theta_d_half_rad = theta_d_rad / 2

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
def sigma_to_variance(sigma, waist):
    variance = waist/2*sigma
    return variance

# calculate the modified fracton of collected power over the receiving aparture when there is no pointing error
def mod_jitter(mu_x, mu_y, sigma_x, sigma_y, waist, varphi_mod):
    A_0 = transmissivity_0(a, waist)
    varphi_x = sigma_to_variance(sigma_x, waist)
    varphi_y = sigma_to_variance(sigma_y, waist)
    # sigma_mod = approximate_jitter_variance(mu_x, mu_y, sigma_x, sigma_y)
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

# def simple_cn2_profile(h):
#     """ A simple model for Cn^2(h) [m^-2/3] """
#     return 1e-14 * np.exp(-h / 1000) 

def cn2_profile(h, v_wind=21, Cn2_0=1e-15):
    term1 = 0.00594 * (v_wind / 27)**2 * (1e-5 * h)**10 * np.exp(-h / 1000)
    term2 = 2.7e-16 * np.exp(-h / 1500)
    term3 = Cn2_0 * np.exp(-h / 100)
    return term1 + term2 + term3


# Calculate the fading loss value
def fading_loss(gamma, mu_x, mu_y, sigma_x, sigma_y, theta_zen_rad, H_atm, waist, tau_zen, varphi_mod):
    eta_t = transmissivity_etat(tau_zen, theta_zen_rad)
    sigma_R_squared = rytov_variance(len_wave, theta_zen_rad, H_ogs, H_atm, cn2_profile)
    A_mod = mod_jitter(mu_x, mu_y, sigma_x, sigma_y, waist, varphi_mod)
    mu = sigma_R_squared/2 * (1+2*varphi_mod**2)
    term1 = (varphi_mod**2) / (2 * (A_mod * eta_t)**(varphi_mod**2))
    term2 = gamma ** (varphi_mod**2 - 1)
    term3 = erfc((np.log((gamma / (A_mod * eta_t))) + mu) / (np.sqrt(2) * math.sqrt(sigma_R_squared)))
    term4 = np.exp(((sigma_R_squared/2) * varphi_mod**2 * (1 + varphi_mod**2)))
    eta_f = term1 * term2 * term3 * term4
    return eta_f

#==================================================================#
# fading_loss : PDF of beam jitter for γ
# qber_loss   : Transmission efficiency Bit error rate with respect to γ
# h_s         : Satellite's altitude (m)
#==================================================================#
def qner_new_infinite(theta_zen_rad, H_atm, waist, tau_zen, varphi_mod):
    
    params = get_beam_jitter_params(condition='weak', theta_d_rad=theta_d_rad)
    mu_x = params["mu_x"]
    mu_y = params["mu_y"]
    sigma_x = params["sigma_x"]
    sigma_y = params["sigma_y"]

    def integrand(gamma_mean):
        return fading_loss(gamma_mean, mu_x, mu_y, sigma_x, sigma_y, theta_zen_rad, H_atm, waist, tau_zen, varphi_mod) * qber_loss(gamma_mean, n_s)

    result, _ = quad(integrand, 0, 1, limit=100, epsabs=1e-9, epsrel=1e-9)
    return result


def weather_condition(tau_zen):
    if tau_zen == 0.91:
        return 'Clear sky'
    elif tau_zen == 0.85:
        return 'Slightly hazy'
    elif tau_zen == 0.75:
        return 'Noticeably hazy'
    else:
        return 'Poor visibility'


def main():
    tau_zen_list = [0.91, 0.85, 0.75, 0.65]
    theta_zen_deg_list = np.linspace(-60, 60, 200)

    plt.figure(figsize=(10, 6))

    for tau_zen in tau_zen_list:
        qber_values = []

        for theta_zen_deg in theta_zen_deg_list:
            theta_zen_rad = math.radians(theta_zen_deg)
            H_atm = 20000 * math.cos(theta_zen_rad)

            waist = beam_waist(h_s, H_ogs, theta_zen_rad, theta_d_rad)
            qber = qner_new_infinite(theta_zen_rad, H_atm, waist, tau_zen, varphi_mod)
            qber_values.append(qber*100)

        label = weather_condition(tau_zen) + f" (τ = {tau_zen})"
        plt.plot(theta_zen_deg_list, qber_values, label=label)

    # グラフ装飾
    plt.xlabel(r"Zenith angle $\theta_{\mathrm{zen}}$ [deg]", fontsize=20)
    plt.ylabel(r"QBER (Quantum Bit Error Rate) [%]", fontsize=20)
    plt.title("QBER vs Zenith Angle for Various Weather Conditions", fontsize=20)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()

    output_path = os.path.join(os.path.dirname(__file__), 'qber_vs_zenith_all_conditions.png')
    plt.savefig(output_path)
    print(f"✅ Saved as: {output_path}")
    plt.show()



if __name__ == "__main__":
    main()



