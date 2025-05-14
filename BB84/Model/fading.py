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
# a = 0.07920
a = 0.75
# a = 0.11672

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
h_s = 550e3  # 550 km

# #==================================================================#
# # tau_zen : Transmission efficiency at zenith
# #==================================================================#
tau_zen = 0.91  # 天頂方向での大気透過率

# #==================================================================#
# # theta_zen_rad : Zenith angle (rad)
# #==================================================================#
theta_zen_rad = math.radians(20)

# #==================================================================#
# # theta_d_rad : Optical beam divergence angle (rad)
# #==================================================================#
theta_d_rad =10e-6 

#==================================================================#
# theta_d_half_rad: beam divergence half-angle
#==================================================================#
theta_d_half_rad = theta_d_rad / 2

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
# Hufnagel-Valley model
#==================================================================#
def Cn2_profile(h, v_wind=21, Cn2_0=1e-13):
    term1 = 0.00594 * (v_wind / 27)**2 * (1e-5 * h)**10 * np.exp(-h / 1000)
    term2 = 2.7e-16 * np.exp(-h / 1500)
    term3 = Cn2_0 * np.exp(-h / 100)
    return 0.2*term1 + term2 + term3

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


#==================================================================#
# calculate the ratios between the equivalent beam-width and (modified) beam-jitter variances
#==================================================================#
def sigma_to_variance(sigma, w_Leq):
    variance = w_Leq/2*sigma
    return variance



def to_decimal_string(x, precision=100):
    if x == 0:
        return "0." + "0" * precision
    return format(x, f'.{precision}f').rstrip('0').rstrip('.')

def main():
    # beam propagation distance
    LoS = satellite_ground_distance(h_s, H_g, theta_zen_rad)
    print(LoS)
    mu_y = 0
    mu_x = 0
    angle_sigma_x = 3e-6
    angle_sigma_y = 3e-6
    sigma_x = angle_sigma_x * LoS
    sigma_y = angle_sigma_y * LoS

    w_L = compute_w_L(lambda_, theta_d_half_rad, LoS, H_atm, H_g, theta_zen_rad, Cn2_profile)

    sigma_mod = compute_sigma_mod(mu_x, mu_y, sigma_x, sigma_y)

    w_Leq_squared = equivalent_beam_width_squared(a, w_L)
    w_Leq = math.sqrt(w_Leq_squared)

    # varphi = varphi_mod(w_Leq_squared, sigma_mod)
    varphi_mod = sigma_to_variance(sigma_mod, w_Leq)

    print(f"Aparture radius:                  {a} [m]")
    print(f"Receiver's Beam width:            {w_L:.3e} [m]")
    print(f"Equivalent Beam width:            {math.sqrt(w_Leq_squared):.3e} [m]")
    print(f"sigma_mod:                        {sigma_mod:.3e} [m]")
    print(f"w_Leq_squared:                    {w_Leq_squared:.3e} [m^2]")
    print(f"varphi_mod:                       {varphi_mod:.3f}")


if __name__ == '__main__':
    main()