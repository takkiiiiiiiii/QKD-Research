from scipy.integrate import quad
import math
import numpy as np
from fading import fading_loss
from qber import qber_loss
from fading import get_beam_jitter_params
from circle_beam_transmissivity import transmissivity_etab, satellite_ground_distance, beam_waist
from atmospheric_transmissivity import transmissivity_etat


a = 0.75
h_s = 500e3
H_a = 0.01
r = 0
n_s = 0.1
#==================================================================#
# theta_d_rad : Optical beam divergence angle (rad)
#==================================================================#
theta_d_rad = 10e-6 
tau_zen = 0.91
condition = 'weak'
theta_zen_deg = 10
theta_zen_rad = math.radians(theta_zen_deg)

waist = beam_waist(h_s, H_a, theta_zen_rad, theta_d_rad)
eta_b = transmissivity_etab(a, r, waist)


#==================================================================#
# fading_loss : PDF of beam jitter for γ
# qber_loss   : Transmission efficiency Bit error rate with respect to γ
#==================================================================#
def qner_ave_infinite():
    params = get_beam_jitter_params(condition=condition, theta_d_rad=theta_d_rad)
    mu_x = params["mu_x"]
    mu_y = params["mu_y"]
    sigma_x = params["sigma_x"]
    sigma_y = params["sigma_y"]

    def integrand(gamma_mean):
        return fading_loss(gamma_mean, mu_x, mu_y, sigma_x, sigma_y) * qber_loss(gamma_mean, n_s)

    result, _ = quad(integrand, 0, 1, limit=100, epsabs=1e-9, epsrel=1e-9)
    return result

def main():
    print(f'QBER: {qner_ave_infinite() * 100} %')

if __name__ == "__main__":
    main()