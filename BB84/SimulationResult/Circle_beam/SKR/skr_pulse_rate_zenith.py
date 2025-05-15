import math
import numpy as np
from scipy.integrate import quad
from scipy.special import erfc
from scipy.special import erf
from scipy.stats import lognorm
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import ScalarFormatter
from scipy.integrate import quad
from secretkeyrate import compute_secret_keyrate
import os


#=======================================================#
#                 Fading Parameters                     #
#=======================================================#
#==================================================================#
# a : Aperture of radius (Receiver radis in meters) (m)
#==================================================================#
# a = 0.07920
a = 0.75

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
H_atm = 20000

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
# mu_x = 0
# mu_y = 0

#==================================================================#
# angle_sigma_x, angle_sigma_y: Beam jitter standard deviations of the Gaussian-distibution jitters (rad)
#==================================================================#
# angle_sigma_x = theta_d_half_rad /2
# angle_sigma_y = theta_d_half_rad /2
# angle_sigma_x = theta_d_half_rad / 3
# angle_sigma_y = theta_d_half_rad / 3

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

#=======================================================#
# Final key settings
#=======================================================#
    #=====================#
    # sifting_coefficient: sifting efficiency
    # p: Parameter estimation coefficient
    # f: Key reconciliation efficiency 
    #======================#i
sifting_coefficient = 0.5
p_estimation = 0.75
kr_efficiency = 1.22


def transmissivity_etal(tau_zen, theta_zen_rad):
    tau_atm = tau_zen ** (1 / math.cos(theta_zen_rad))
    return tau_atm

#==================================================================#
# I_a: random intensity loss due to the atmospheric turbulence
#==================================================================#
def compute_intensity_loss(sigma_R_squared, size=1):
    shape_param = np.sqrt(sigma_R_squared)
    
    I_a_random = lognorm.rvs(shape_param, loc=0, scale=1, size=size)
    
    return I_a_random
    

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
def compute_varphi_mod(sigma, w_Leq):
    variance = w_Leq/(2*sigma)
    return variance

def maxmum_fracrion_A0(w_L):
    nu = (math.sqrt(math.pi) * a) / (math.sqrt(2) * w_L)

    A_0 = erf(nu)**2
    return A_0

# calculate the modified fracton of collected power over the receiving aparture when there is no pointing error
def mod_jitter(mu_x, mu_y, sigma_x, sigma_y, w_L, w_Leq):
    A_0 = maxmum_fracrion_A0(w_L)

    varphi_x = compute_varphi_mod(sigma_x, w_Leq)
    varphi_y = compute_varphi_mod(sigma_y, w_Leq)
    sigma_mod = approximate_jitter_variance(mu_x, mu_y, sigma_x, sigma_y)
    varphi_mod = compute_varphi_mod(sigma_mod, w_Leq)
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

def cn2_profile(h, v_wind=21, Cn2_0=1e-15):
    term1 = 0.00594 * (v_wind / 27)**2 * (1e-5 * h)**10 * np.exp(-h / 1000)
    term2 = 2.7e-16 * np.exp(-h / 1500)
    term3 = Cn2_0 * np.exp(-h / 100)
    return term1 + term2 + term3


# Calculate the fading loss value
def fading_loss(gamma, mu_x, mu_y, sigma_x, sigma_y, theta_zen_rad, H_atm, w_L,w_Leq, tau_zen):
    eta_t = transmissivity_etal(tau_zen, theta_zen_rad)
    sigma_R_squared = rytov_variance(lambda_, theta_zen_rad, H_g, H_atm, cn2_profile)
    sigma_mod = approximate_jitter_variance(mu_x, mu_y, sigma_x, sigma_y)
    varphi_mod = compute_varphi_mod(sigma_mod, w_Leq)
    A_mod = mod_jitter(mu_x, mu_y, sigma_x, sigma_y, w_L, w_Leq)
    mu = sigma_R_squared/2 * (1+(2*varphi_mod**2))
    term1 = (varphi_mod**2) / (2 * (A_mod * eta_t)**(varphi_mod**2))
    term2 = gamma ** (varphi_mod**2 - 1)
    term3 = erfc((np.log((gamma / (A_mod * eta_t))) + mu) / (np.sqrt(2) * math.sqrt(sigma_R_squared)))
    term4 = np.exp(((sigma_R_squared/2) * varphi_mod**2 * (1 + varphi_mod**2)))
    eta_f = term1 * term2 * term3 * term4
    return eta_f


#=======================================================#
# Beam waist function
#=======================================================#
def beam_waist(h_s, H_g, theta_zen_rad, theta_d_half_rad):
    L_a = satellite_ground_distance(h_s, H_g, theta_zen_rad)
    waist = L_a * theta_d_half_rad
    return waist

#==================================================================#
# Compute beam propagation disteance
#==================================================================#
def satellite_ground_distance(h_s, H_g, theta_zen_rad):
    return (h_s - H_g) / math.cos(theta_zen_rad)


def equivalent_beam_width_squared(a, w_L):
    # w_L: beam radius at receiver before aperture clipping
    nu = (math.sqrt(math.pi) * a) / (math.sqrt(2) * w_L)
    numerator = math.sqrt(math.pi) * erf(nu)
    denominator = 2 * nu * math.exp(-nu**2)
    return w_L**2 * (numerator / denominator)


def qner_new_infinite(theta_zen_rad, H_atm, w_L, tau_zen, LoS):
    mu_x = 0
    mu_y = 0
    sigma_x = theta_d_half_rad /5 * LoS
    sigma_y = theta_d_half_rad /5 * LoS

    w_Leq_squared = equivalent_beam_width_squared(a, w_L)
    w_Leq = math.sqrt(w_Leq_squared)

    def integrand(eta):
        return fading_loss(eta, mu_x, mu_y, sigma_x, sigma_y, theta_zen_rad, H_atm, w_L, w_Leq, tau_zen) * qber_loss(eta, n_s)
    
    def integrand2(eta):
        return fading_loss(eta, mu_x, mu_y, sigma_x, sigma_y, theta_zen_rad, H_atm, w_L, w_Leq, tau_zen)* parameter_q(eta, n_s)

    result, _ = quad(integrand, 0, np.inf, limit=100, epsabs=1e-9, epsrel=1e-9)
    result2, _ = quad(integrand2, 0, np.inf, limit=100, epsabs=1e-9, epsrel=1e-9)
    return result, result2

#==================================================================#
#  Compute QBER estimation
#==================================================================#
def qber_loss(eta, n_s):
    denominator =  e_0 * (Y_0*(1+P_AP)) + (e_pol+e_0*P_AP) * (1-np.exp(-n_s*eta))
    numerator = (Y_0*(1+P_AP)) + (1-np.exp(-n_s*eta)) * (1+P_AP)
    qber = denominator/numerator
    return qber

def parameter_q(eta, n_s):
    param_q = (Y_0*(1+P_AP)) + (1-np.exp(-n_s*eta)) * (1+P_AP)
    return param_q

def weather_condition(tau_zen):
    if tau_zen == 0.91:
        return 'Clear sky'
    elif tau_zen == 0.85:
        return 'Slightly hazy'
    elif tau_zen == 0.75:
        return 'Noticeably hazy'
    elif tau_zen == 0.53:
        return 'Poor visibility'
    else:
        return 'Unknown condition'
    

def main():
    tau_zen_list = [0.91, 0.85, 0.75, 0.53]
    theta_zen_deg_list = np.linspace(-60, 60, 100)
    plt.figure(figsize=(10, 6))

    for tau_zen in tau_zen_list:
        skr_pulse_list = []

        # Get weather condition and H_atm from tau_zen
        weather_condition_str = weather_condition(tau_zen)
        
        for theta_zen_deg in theta_zen_deg_list:
            if theta_zen_deg < 0:
                theta_zen_rad = np.radians(-theta_zen_deg)
            else:
                theta_zen_rad = np.radians(theta_zen_deg)
            # print(theta_zen_deg)
            LoS = satellite_ground_distance(h_s, H_g, theta_zen_rad)
            waist = beam_waist(h_s, H_g, theta_zen_rad, theta_d_half_rad)
            qber, param_q = qner_new_infinite(theta_zen_rad, H_atm, waist, tau_zen, LoS)
            # qber = qner_new_infinite(theta_zen_rad, H_atm, waist, tau_zen, LoS)
        
           
            # prob_error = qber_loss(insta_eta, n_s)
            secret_keyrate = compute_secret_keyrate(qber, param_q, sifting_coefficient, p_estimation, kr_efficiency)
            skr_pulse = secret_keyrate / 1e-4
            skr_pulse_list.append(skr_pulse)

        label = weather_condition_str + f" (τ = {tau_zen})"
        plt.plot(theta_zen_deg_list, skr_pulse_list, label=label)
        

    # plt.xlabel("Zenith angle (degrees)", fontsize=20)
    # plt.ylabel("Secret Keyrate(bit/pulse)", fontsize=20)
    # plt.title("Secret Keyrate vs Zenith Angle" + f" $(n_s = {n_s})$", fontsize=20)
    plt.xlabel(r"Zenith angle (degrees)", fontsize=20)
    plt.ylabel(r"Secret key rate ($\mathrm{bit/pulse}$), from bps $\div 10^4$", fontsize=16)
    plt.title(r"Secret Key Rate per Pulse ($\mathrm{bit/pulse}$) vs Zenith Angle" + f" $(n_s = {n_s})$", fontsize=16)
    plt.grid(True)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend()
    ax = plt.gca() 
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-3, 3))  # 指数表示の範囲設定（例: 1e-3 〜 1e+3）
    ax.yaxis.set_major_formatter(formatter)
    ax.yaxis.get_offset_text().set_fontsize(20)
    output_path = os.path.join(os.path.dirname(__file__), f'skr_per_pulse_vs_zenith_all_conditions_{n_s}.png')
    plt.savefig(output_path)
    print(f"✅ Saved as: {output_path}")
    plt.show()



if __name__ == "__main__":
    main()
