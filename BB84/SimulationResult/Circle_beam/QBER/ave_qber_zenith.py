import math
import numpy as np
from scipy.integrate import quad
from scipy.special import erfc
from scipy.special import erf
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
# n_s : average number of photon
#==================================================================#
n_s = 0.1

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
h_s = 500e3  # 500 km

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



def satellite_ground_distance(h_s, H_g, theta_zen_rad):
    return (h_s - H_g) / math.cos(theta_zen_rad)

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


def cn2_profile(h, v_wind=21, Cn2_0=1e-13):
    term1 = 0.00594 * (v_wind / 27)**2 * (1e-5 * h)**10 * np.exp(-h / 1000)
    term2 = 2.7e-16 * np.exp(-h / 1500)
    term3 = Cn2_0 * np.exp(-h / 100)
    return term1 + term2 + term3


# Calculate the fading loss value
def fading_loss(gamma, mu_x, mu_y, sigma_x, sigma_y, theta_zen_rad, H_atm, w_Leq, tau_zen, varphi_mod):
    eta_t = transmissivity_etat(tau_zen, theta_zen_rad)
    sigma_R_squared = rytov_variance(lambda_, theta_zen_rad, H_g, H_atm, cn2_profile)
    A_mod = mod_jitter(mu_x, mu_y, sigma_x, sigma_y, w_Leq, varphi_mod)
    mu = sigma_R_squared/2 * (1+2*varphi_mod**2)
    term1 = (varphi_mod**2) / (2 * (A_mod * eta_t)**(varphi_mod**2))
    term2 = gamma ** (varphi_mod**2 - 1)
    term3 = erfc((np.log((gamma / (A_mod * eta_t))) + mu) / (np.sqrt(2) * math.sqrt(sigma_R_squared)))
    term4 = np.exp(((sigma_R_squared/2) * varphi_mod**2 * (1 + varphi_mod**2)))
    eta_f = term1 * term2 * term3 * term4
    return eta_f


def weather_condition(tau_zen):
    if tau_zen == 0.91:
        return 'Clear sky'
    elif tau_zen == 0.85:
        return 'Slightly hazy'
    elif tau_zen == 0.75:
        return 'Noticeably hazy'
    else:
        return 'Poor visibility'


def Cn_squared(h):
    return 1e-13

# Beam footprint radius at receiver including turbulence
def compute_w_L(lambda_, theta_d_half_rad, L, H_atm, H_OGS, theta_zen_rad):
    k = 2 * math.pi / lambda_

    w_0 = lambda_ / (math.pi * theta_d_half_rad)

    W = w_0 * math.sqrt(1 + (2 * L) / (k * w_0))

    def integrand(h):
        return Cn_squared(h) * ((h - H_OGS) / (H_atm - H_OGS))**(5/3)

    # integral_result, _ = quad(integrand, H_OGS, H_atm)
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
def sigma_to_variance(sigma, w_Leq):
    variance = w_Leq/2*sigma
    return variance

# calculate the modified fracton of collected power over the receiving aparture when there is no pointing error
def mod_jitter(mu_x, mu_y, sigma_x, sigma_y, w_Leq):
    A_0 = transmissivity_0(a, w_Leq)
    varphi_x = sigma_to_variance(sigma_x, w_Leq)
    varphi_y = sigma_to_variance(sigma_y, w_Leq)
    sigma_mod = approximate_jitter_variance(mu_x, mu_y, sigma_x, sigma_y)
    varphi_mod = sigma_to_variance(sigma_mod, w_Leq)
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


def cn2_profile(h, v_wind=21, Cn2_0=1e-13):
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
def equivalent_beam_width_squared(a, w_Leq):
    nu = (math.sqrt(math.pi) * a) / (math.sqrt(2) * w_Leq)
    numerator = math.sqrt(math.pi) * erf(nu)
    denominator = 2 * nu * math.exp(-nu**2)
    w_Leq_squared = w_Leq**2 * (numerator / denominator)
    return w_Leq_squared

# Calculate varphi_mod
def compute_varphi_mod(w_Leq_squared, sigma_mod):
    w_Leq = math.sqrt(w_Leq_squared)
    return w_Leq / (2 * sigma_mod)

# Calculate the fading loss value
def fading_loss(gamma, mu_x, mu_y, sigma_x, sigma_y, theta_zen_rad, H_atm, w_Leq, tau_zen, varphi_mod):
    eta_t = transmissivity_etat(tau_zen, theta_zen_rad)
    sigma_R_squared = rytov_variance(lambda_, theta_zen_rad, H_g, H_atm, cn2_profile)
    varphi_mod = 4.3292
    A_mod = mod_jitter(mu_x, mu_y, sigma_x, sigma_y, w_Leq)
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
def qner_new_infinite(theta_zen_rad, H_atm, w_L, tau_zen, LoS):
    # beam propagation distance
    # mu_y = 0
    # mu_x = 0
    # angle_sigma_x = 3e-6
    # angle_sigma_y = 3e-6
    sigma_x = angle_sigma_x * LoS
    sigma_y = angle_sigma_y * LoS

    sigma_mod = compute_sigma_mod(mu_x, mu_y, sigma_x, sigma_y)
    w_Leq_squared = equivalent_beam_width_squared(a, w_L)
    varphi_mod = compute_varphi_mod(w_Leq_squared, sigma_mod)
    w_Leq = math.sqrt(w_Leq_squared)
    

    def integrand(gamma_mean):
        return fading_loss(gamma_mean, mu_x, mu_y, sigma_x, sigma_y, theta_zen_rad, H_atm, w_Leq, tau_zen, varphi_mod) * qber_loss(gamma_mean, n_s)

    result, _ = quad(integrand, 0, np.inf, limit=100, epsabs=1e-9, epsrel=1e-9)
    return result


def main():
    tau_zen_list = [0.91, 0.85, 0.75, 0.65]
    theta_zen_deg_list = np.linspace(-60, 60, 200)

    plt.figure(figsize=(10, 6))

    for tau_zen in tau_zen_list:
        qber_values = []

        for theta_zen_deg in theta_zen_deg_list:
            theta_zen_rad = math.radians(theta_zen_deg)
            H_atm = 20000
            LoS = satellite_ground_distance(h_s, H_g, theta_zen_rad)
            w_L = compute_w_L(lambda_, theta_d_half_rad, LoS, H_atm, H_g, theta_zen_rad)

            qber = qner_new_infinite(theta_zen_rad, H_atm, w_L, tau_zen, LoS)
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