import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import os, sys

# モジュール読み込み
simulation_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../', 'Model'))
sys.path.append(simulation_path)

from circle_beam_transmissivity import transmissivity_etab, satellite_ground_distance, beam_waist
from atmospheric_transmissivity import transmissivity_etat
from qber import qber_loss
from fading import get_beam_jitter_params, fading_loss


# 定数
tau_zen = 0.91
a = 0.75
G = 6.67430e-11
M_T = 5.972e24
D_E = 6378e3
h_s = 500e3
H_a = 0.01
theta_d_rad = 20e-6


def qner_new_infinite(eta_b, eta_t):
    params = get_beam_jitter_params(condition="strong", theta_d_rad=theta_d_rad)
    mu_x = params["mu_x"]
    mu_y = params["mu_y"]
    sigma_x = params["sigma_x"]
    sigma_y = params["sigma_y"]

    def integrand(gamma):
        return fading_loss(gamma, mu_x, mu_y, sigma_x, sigma_y) * qber_loss(gamma)

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
    r = 0  # displacement 固定
    theta_zen_deg_list = np.linspace(-60, 60, 200)
    qber_values = []

    for theta_zen_deg in theta_zen_deg_list:
        theta_zen_rad = math.radians(theta_zen_deg)

        # チャネルパラメータ計算
        L_a = satellite_ground_distance(h_s, H_a, theta_zen_rad)
        waist = beam_waist(h_s, H_a, theta_zen_rad, theta_d_rad)

        eta_t = transmissivity_etat(tau_zen, theta_zen_rad)
        eta_b = transmissivity_etab(a, r, waist)

        qber = qner_new_infinite(eta_b, eta_t)
        qber_values.append(qber)

    # グラフ描画
    weather = weather_condition(tau_zen)
    plt.figure(figsize=(10, 6))
    plt.plot(theta_zen_deg_list, qber_values, label=f"r = {r} m")
    plt.xlabel(r"Zenith angle $\theta_{\mathrm{zen}}$ [deg]", fontsize=18)
    plt.ylabel(r"QBER (Quantum Bit Error Rate)", fontsize=18)
    plt.title(fr"QBER vs Zenith Angle ({weather})", fontsize=20)
    plt.grid(True)
    plt.legend(fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()

    output_path = os.path.join(os.path.dirname(__file__), f'qber_vs_zenith_{weather}.png')
    plt.savefig(output_path)
    print(f"✅ Saved as: {output_path}")
    plt.show()


if __name__ == "__main__":
    main()

