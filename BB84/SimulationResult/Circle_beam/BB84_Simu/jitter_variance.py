from scipy.special import erf
import math

a = 0.75

def modified_beam_jitter_variance(mu_x, mu_y, sigma_x, sigma_y):
    """
    Calculate the modified beam-jitter variance approximation.

    Parameters:
        mu_x (float): Mean position of the beam in the x-direction
        mu_y (float): Mean position of the beam in the y-direction
        sigma_x (float): Standard deviation of the beam in the x-direction
        sigma_y (float): Standard deviation of the beam in the y-direction

    Returns:
        float: Modified beam-jitter variance (σ_mod)
    """
    numerator = (3 * mu_x**2 * sigma_x**4 +
                 3 * mu_y**2 * sigma_y**4 +
                 sigma_x**6 +
                 sigma_y**6)
    sigma_mod = (numerator / 2)**(1/3)
    return sigma_mod

def to_decimal_string(x, precision=70):
    if x == 0:
        return "0." + "0" * precision
    return format(x, f'.{precision}f').rstrip('0').rstrip('.')

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

theta_d_half_rad = 10e-6 
mu_x = 0
mu_y = 0
sigma_x = theta_d_half_rad/5
sigma_y = theta_d_half_rad/5


def sigma_to_variance(sigma, w_Leq):
    variance = w_Leq/(2*sigma)
    return variance





sigma_mod = modified_beam_jitter_variance(mu_x, mu_y, sigma_x, sigma_y)
sigma_mod2 = approximate_jitter_variance(mu_x, mu_y, sigma_x, sigma_y)


w_L = 10.999799999999999
w_Leq =11.026621099609564
varphi = sigma_to_variance(sigma_mod, w_Leq)
print(f'varphi: {varphi}')

print(f"Modified beam-jitter variance σ_mod: {to_decimal_string(sigma_mod)}")
print(f"Modified beam-jitter variance σ_mod: {to_decimal_string(sigma_mod2)}")



