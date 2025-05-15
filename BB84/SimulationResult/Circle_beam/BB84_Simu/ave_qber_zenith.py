import math
import numpy as np
from scipy.integrate import quad
from scipy.special import erfc
from scipy.special import erf
from scipy.stats import lognorm
import matplotlib.pyplot as plt
from scipy.integrate import quad
import os, sys

#=======================================================#
#                 Fading Parameters                     #
#=======================================================#
#==================================================================#
# a : Aperture of radius (Receiver radis in meters) (m)
#==================================================================#
# a = 0.07920
a = 0.75

# r = 5
#==================================================================#
# n_s : average number of photon
#==================================================================#
n_s = 0.8

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
h_s = 550e3  # 500 km

#==================================================================#
# H_a : Upper end of atmosphere (km)
#==================================================================#
H_atm = 200000

#==================================================================#
# theta_d_rad : Optical beam divergence angle (rad)
#==================================================================#
theta_d_rad = 20e-6 

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
# angle_sigma_x = theta_d_half_rad /2
# angle_sigma_y = theta_d_half_rad /2
angle_sigma_x = theta_d_half_rad / 3
angle_sigma_y = theta_d_half_rad / 3

#=======================================================#
# QBER parameters
#=======================================================#
    #=====================#
    # n_s   : average numher of photon from Alice
    # e_0   : the error rate of the background
    # Y_0   : the background rate which includes the detector dark count and other background contributions
    # P_pa  : After-pulsing probability
    # e_pol : Probability of the polarisation errors
    #======================#i
e_0 = 0.5
Y_0 = 1e-4
P_AP = 0.02
e_pol = 0.033

#==================================================================#
# I_a: random intensity loss due to the atmospheric turbulence
#==================================================================#
def compute_intensity_loss(sigma_R_squared, size=1):
    shape_param = np.sqrt(sigma_R_squared)
    
    I_a_random = lognorm.rvs(shape_param, loc=0, scale=1, size=size)
    
    return I_a_random
    


#==================================================================#
# calculate modified beam-jitter variance approximation
#==================================================================#
def approximate_jitter_variance(mu_x, mu_y, sigma_x, sigma_y):
    numerator = (
        3 * mu_x**2 * sigma_x**4 +
        3 * mu_y**2 * sigma_y**4 +
        sigma_x**6 +
        sigma_y**6
    )
    sigma_mod_value = (numerator / 2) ** (1/3)
    return sigma_mod_value


#==================================================================#
# Compute beam propagation disteance
#==================================================================#
def satellite_ground_distance(h_s, H_g, theta_zen_rad):
    return (h_s - H_g) / math.cos(theta_zen_rad)


def transmissivity_etal(tau_zen, theta_zen_rad):
    tau_atm = tau_zen ** (1 / math.cos(theta_zen_rad))
    return tau_atm


def fading_loss(eta, mu_x, mu_y, sigma_x, sigma_y, theta_zen_rad, H_atm, w_L, w_Leq, tau_zen, varphi_mod):
    eta_l = transmissivity_etal(tau_zen, theta_zen_rad)
    sigma_R_squared = rytov_variance(lambda_, theta_zen_rad, H_g, H_atm, Cn2_profile)
    # print(f'sigma_R_squared : {sigma_R_squared}')

    A_mod = mod_jitter(mu_x, mu_y, sigma_x, sigma_y, w_L, w_Leq)
    mu = sigma_R_squared / 2 * (1 + 2 * varphi_mod**2)
    
    term1 = (varphi_mod**2) / (2 * (A_mod * eta_l)**(varphi_mod**2))
    term2 = eta ** (varphi_mod**2-1)
    term3 = erfc((np.log((eta / (A_mod * eta_l))) + mu) / (np.sqrt(2) * math.sqrt(sigma_R_squared)))
    
    # Check for overflow possibility in term4
    exponent = (sigma_R_squared / 2) * varphi_mod**2 * (1 + varphi_mod**2)
    # print(f"Exponent for term4: {exponent}")
    
    if exponent > 700:  # Avoid overflow
        print(f"Warning: Exponent is too large, term4 will overflow. Exponent value: {exponent}")
    
    term4 = np.exp(np.clip(exponent, None, 700))  # Limit the exponent to avoid overflow
    # print(exponent)
    eta_f = term1 * term2 * term3 * term4
    return eta_f

#==================================================================#
# Beam footprint radius at receiver including turbulence
#==================================================================#
def compute_w_L(lambda_, theta_d_half_rad, L, H_atm, H_OGS, theta_zen_rad, Cn2_profile):
    k = 2 * math.pi / lambda_

    w_0 = lambda_ / (math.pi * theta_d_half_rad)

    W = w_0 * math.sqrt(1 + (2 * L) / (k * w_0**2))

    def integrand(h):
        return Cn2_profile(h) * ((h - H_OGS) / (H_atm - H_OGS))**(5/3)

    # integral_result, _ = quad(integrand, H_OGS, H_atm)
    integral_result, _ = quad(integrand, H_OGS, H_atm)


    T = 4.35 * ((2 * L) / (k * W**2))**(5/6) * \
        k**(7/6) * (H_atm - H_OGS)**(5/6) * \
        (1 / math.cos(theta_zen_rad))**(11/6) * integral_result

    # Step 5: compute final beam radius at receiver
    w_L = W * math.sqrt(1 + T)
    return w_L


#==================================================================#
# calculate the ratios between the equivalent beam-width and (modified) beam-jitter variances
#==================================================================#
def sigma_to_variance(sigma, w_Leq):
    variance = w_Leq/2*sigma
    return variance


#==================================================================#
# calculate the modified fracton of collected power over the receiving aparture when there is no pointing error
#==================================================================#
def mod_jitter(mu_x, mu_y, sigma_x, sigma_y, w_L, w_Leq):
    A_0 = maxmum_fracrion_A0(w_L)
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


#==================================================================#
# Compute Rytov variance σ_R^2 for atmospheric turbulence
#==================================================================#
def rytov_variance(len_wave, theta_zen_rad, H_OGS, H_atm, Cn2_profile):
    k = 2 * np.pi / len_wave
    sec_zenith = 1 / np.cos(theta_zen_rad)

    def integrand(h):
        return Cn2_profile(h) * (h - H_OGS)**(5/6)

    integral, _ = quad(integrand, H_OGS, H_atm, limit=100, epsabs=1e-9, epsrel=1e-9)

    sigma_R_squared = 2.25 * (k)**(7/6) * sec_zenith**(11/6) * integral

    return sigma_R_squared


#==================================================================#
# Hufnagel-Valley model
#==================================================================#
def Cn2_profile(h, v_wind=21, Cn2_0=1e-13):
    term1 = 0.00594 * (v_wind / 27)**2 * (1e-5 * h)**10 * np.exp(-h / 1000)
    term2 = 2.7e-16 * np.exp(-h / 1500)
    term3 = Cn2_0 * np.exp(-h / 100)
    return term1 + term2 + term3


#==================================================================#
# Calculate sigma_mod
#==================================================================#
def compute_sigma_mod(mu_x, mu_y, sigma_x, sigma_y):
    numerator = (
        3 * mu_x**2 * sigma_x**4 +
        3 * mu_y**2 * sigma_y**4 +
        sigma_x**6 +
        sigma_y**6
    )
    sigma_mod = (numerator / 2) ** (1/3)
    return sigma_mod

#==================================================================#
# Compute equivalent Beam Width
#==================================================================#
def equivalent_beam_width_squared(a, w_L):
    # w_L: beam radius at receiver before aperture clipping
    nu = (math.sqrt(math.pi) * a) / (math.sqrt(2) * w_L)
    numerator = math.sqrt(math.pi) * erf(nu)
    denominator = 2 * nu * math.exp(-nu**2)
    return w_L**2 * (numerator / denominator)

def beam_waist(h_s, H_g, theta_zen_rad, theta_d_half_rad):
    L_a = satellite_ground_distance(h_s, H_g, theta_zen_rad)
    waist = L_a * theta_d_half_rad
    return waist

#==================================================================#
# Calculate varphi_mod
#==================================================================#
def compute_varphi_mod(w_Leq_squared, sigma_mod):
    w_Leq = math.sqrt(w_Leq_squared)
    return w_Leq / (2 * sigma_mod)


#==================================================================#
#  Compute QBER estimation
#==================================================================#
def qber_loss(eta, n_s):
    denominator =  e_0 * (Y_0*(1+P_AP)) + (e_pol+e_0*P_AP) * (1-np.exp(-n_s*eta))
    numerator = (Y_0*(1+P_AP)) + (1-np.exp(-n_s*eta)) * (1+P_AP)
    qber = denominator/numerator
    return qber


#==================================================================#
# Compute eta_p : beam misalignment and spreading loss
#==================================================================#
def transmissivity_etap(theta_zen_rad, r):
    LoS = satellite_ground_distance(h_s, H_g, theta_zen_rad)

    # w_L = compute_w_L(lambda_, theta_d_half_rad, LoS, H_atm, H_g, theta_zen_rad, Cn2_profile)
    w_L = beam_waist(h_s, H_g, theta_zen_rad, theta_d_half_rad)
    # print(fr'w_L: {w_L}')
    # w_Leq_squared = equivalent_beam_width_squared(a, w_L)
    nu = (math.sqrt(math.pi) * a) / (math.sqrt(2) * w_L)
    # print(fr'nu: {nu}')
    A0 = erf(nu)**2
    # print(f'A0: {A0}')
    eta_p = A0 * np.exp(-(2*r**2)/(w_L))
    return eta_p


#==================================================================#
# Compute A_0 : maximum fraction of received power
#==================================================================#
def maxmum_fracrion_A0(w_L):
    nu = (math.sqrt(math.pi) * a) / (math.sqrt(2) * w_L)

    A_0 = erf(nu)**2
    return A_0


#==================================================================#
# fading_loss : PDF of beam jitter for γ
# qber_loss   : smission efficiency Bit error rate with respect to γ
# h_s         : Satellite's altitude (m)
#==================================================================#
def qner_new_infinite(theta_zen_rad, H_atm, w_L, tau_zen, LoS):
    sigma_x = angle_sigma_x * LoS
    sigma_y = angle_sigma_y * LoS

    w_Leq_squared = equivalent_beam_width_squared(a, w_L)
    w_Leq = math.sqrt(w_Leq_squared)
    sigma_mod = compute_sigma_mod(mu_x, mu_y, sigma_x, sigma_y)
    varphi_mod = sigma_to_variance(sigma_mod, w_Leq)

    def integrand(eta):
        return fading_loss(eta, mu_x, mu_y, sigma_x, sigma_y, theta_zen_rad, H_atm, w_L, w_Leq, tau_zen, varphi_mod) * qber_loss(eta, n_s)

    result, _ = quad(integrand, 0, np.inf, limit=100, epsabs=1e-9, epsrel=1e-9)
    return result

#==================================================================#
# Compute radial displacement due to beam pointing jitter.
#==================================================================#
def compute_radial_displacement(mu_x, mu_y, angle_sigma_x, angle_sigma_y, LoS, size=1):
    sigma_x = angle_sigma_x * LoS
    sigma_y = angle_sigma_y * LoS

    x = np.random.normal(loc=mu_x, scale=sigma_x, size=size)
    y = np.random.normal(loc=mu_y, scale=sigma_y, size=size)

    # ラジアル変位 [m]
    r = np.sqrt(x**2 + y**2)
    return r

#==================================================================#
# Compute intensity loss due to the atmospheric turbulence
#==================================================================#
def compute_intensity_loss(sigma_R_squared, size=1):
    shape_param = np.sqrt(sigma_R_squared) 
    
    I_a_random = lognorm.rvs(shape_param, loc=0, scale=1, size=size)
    
    return I_a_random

#==================================================================#
# Compute transmissivity condidering the beam misalignemnt and spreading loss 
# , atmospheric attenuation and turbuence
#==================================================================#
def compute_eta(eta_t, sigma_R_squared, theta_zen_rad, lambda_, h_s, H_g, Cn2_profile, a, r, size=1):
    
    # intensity loss due to the beam misalignemnt and spreading loss 
    I_a_random = compute_intensity_loss(sigma_R_squared, size)
    
    # transmissivity considering 
    eta_p = transmissivity_etap(theta_zen_rad, r)
    
    # transmissivity considering atmospheric attenuation
    eta = eta_t * I_a_random * eta_p
    
    return eta

def weather_condition(tau_zen):
    if tau_zen == 0.91:
        return 'Clear sky', 23000  # H_atm for clear sky
    elif tau_zen == 0.85:
        return 'Slightly hazy', 15000  # H_atm for slightly hazy
    elif tau_zen == 0.75:
        return 'Noticeably hazy', 10000  # H_atm for noticeably hazy
    elif tau_zen == 0.53:
        return 'Poor visibility', 5000  # H_atm for poor visibility
    else:
        return 'Unknown condition', 10000  # Default value



def main():
    tau_zen_list = [0.91, 0.85, 0.75, 0.53]
    theta_zen_deg_list = np.linspace(-40, 40, 100)
    plt.figure(figsize=(10, 6))

    for tau_zen in tau_zen_list:
        qber_values = []

        # Get weather condition and H_atm from tau_zen
        weather_condition_str, H_atm = weather_condition(tau_zen)
        
        for theta_zen_deg in theta_zen_deg_list:
            if theta_zen_deg < 0:
                theta_zen_rad = np.radians(-theta_zen_deg)
            else:
                theta_zen_rad = np.radians(theta_zen_deg)
            # print(theta_zen_deg)
            LoS = satellite_ground_distance(h_s, H_g, theta_zen_rad)
            # print(f'LoS: {LoS}')
            # print(h_s)
            # print(H_g)
            # print(theta_zen_rad)
            r = compute_radial_displacement(mu_x, mu_y, angle_sigma_x, angle_sigma_y, LoS, size=1)
            w_L = compute_w_L(lambda_, theta_d_half_rad, LoS, H_atm, H_g, theta_zen_rad, Cn2_profile)
            # print(f'w_L: {w_L}')

            qber = qner_new_infinite(theta_zen_rad, H_atm, w_L, tau_zen, LoS)
            qber_values.append(qber*100)

        label = f"{weather_condition_str} (τ = {tau_zen})"
        plt.plot(theta_zen_deg_list, qber_values, label=label)

    # グラフ装飾
    plt.xlabel(r"Zenith angle $\theta_{\mathrm{zen}}$ [deg]", fontsize=20)
    plt.ylabel(r"Probability Error[%]", fontsize=20)
    plt.title("QBER vs Zenith Angle for Various Weather Conditions", fontsize=20)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()

    # output_path = os.path.join(os.path.dirname(__file__), f'qber_vs_zenith_all_conditions_{n_s}.png')
    # plt.savefig(output_path)
    # print(f"✅ Saved as: {output_path}")
    plt.show()



if __name__ == "__main__":
    main()
