from scipy.special import lambertw, i0, i1
import math
import numpy as np
from scipy.integrate import quad
from circle_beam_transmissivity import transmissivity_0, beam_waist
from atmospheric_transmissivity import transmissivity_etat
from circle_beam_transmissivity import transmissivity_etab


a = 0.75      # Aperture of radius (Receiver radis in meters)
r = 5
ratios = np.arange(0, 3.1, 0.1)
r0 = [r * a for r in ratios]
mag_w1 = [0.2, 1.0, 1.8]
mag_w2 = [0.1, 0.9, 1.7]
chi = [math.pi/3, math.pi/4, math.pi/5]
chi_show = [3, 4, 5]


# Complementary error function
def erfc(x):
    integral, _ = quad(lambda t: np.exp(-t**2), x, np.inf)
    return (2 / np.sqrt(np.pi)) * integral


#=======================================================#
# Fading parameters
#=======================================================#

#=====================#
# varphi_mod   : the ratio between the equivalent beam width and the modified beam-jitter variance
#               (修正されたビーム幅と修正されたジッタ分散の比率)
# len_wave     : optical wavelength 
#               (光波長。通信で使用する光の波長)
# Y_0          : the background rate which includes the detector dark count and other background contributions
#               (検出器のダークカウントやその他の背景ノイズを含む背景率)
# P_pa         : After-pulsing probability
#               (アフターパルス確率。光子検出器で発生するアフターパルスの確率)
# e_pol        : Probability of the polarization errors
#               (偏光誤差の確率。偏光エラーが発生する確率)
#=====================#

varphi_mod = 4.3292
# 修正されたジッタパラメータ。ビームの有効幅と修正されたジッタ分散の比率を表す無次元パラメータ。

A_0 = transmissivity_0()
# 初期透過率。伝送路を通る光の初期透過率を示す。

len_wave = 0.85  # 波長 (um)
# Oprical wavelength. 光波長。通常、FSO（Free-Space Optics）通信などでは、使用する光の波長を指定します。ここでは0.85μmの波長を指定しています。

sigma_R = 1.0
# Rytov分散。大気乱流による光強度の変動を評価する指標。1は中程度の乱流を意味します。

h_s = 500e3  # 衛星の高さ（m）
# 衛星の高度。衛星と地上局間の通信における大気を通過する光の経路長に影響を与える。

H_a = 0.01  # 大気上端高度 (km)
# 大気の上端高度。大気の終わりの地点を示す。

tau_zen = 0.85
# 天頂角の透過率。大気の影響を受ける透過率を示し、ここでは0.85の透過率を指定。

theta_zen_deg = 30
# 天頂角 (度)。地上局から見た衛星の天頂角を度単位で指定。

theta_zen_rad = math.radians(theta_zen_deg)
# 天頂角をラジアン単位に変換。

theta_d_rad = 10e-6  # 発散角（ラジアン）
# Divergence angle
# 光ビームの発散角。ビームが広がる角度の半分を表します。

theta = theta_d_rad / 2
# Divergence half-angle
# 発散角の半分。ビームの半分の発散角を求めています。

rytov = 1
# Rytovのパラメータ。乱流強度の指標として使用され、値が1は中程度の乱流を示します。

mu_x = 0
mu_y = 0
# x方向、y方向の平均オフセット。ここでは0として指定されています。

sigma_x = theta / 5 * h_s
# Standard deviation of beam jitter in the x-direction.
# x方向のビームジッタの標準偏差。ここではビームの発散角と衛星高度に基づいて計算されています。

sigma_y = theta / 5 * h_s
# Standard deviation of beam jitter in the y-direction.
# y方向のビームジッタの標準偏差。x方向と同様に計算されています。

waist = beam_waist(h_s, H_a, theta_zen_rad, theta_d_rad)
# beam waist
# ビームの腰（ウエスト）。ビームが最も狭くなる位置を示します。

varphi_x = waist / (2 * sigma_x)
# Modified beam jitter parameter in x-direction.
# x方向の修正ビームジッタパラメータ。ビームの腰とx方向のビームジッタを基に計算。

varphi_y = waist / (2 * sigma_y)
# Modified beam jitter parameter in y-direction.
# y方向の修正ビームジッタパラメータ。x方向と同様に計算。


def mod_jitter():
    term1 = 1 / (varphi_mod ** 2)
    term2 = 1 / (2 * varphi_x ** 2)
    term3 = 1 / (2 * varphi_y ** 2)
    term4 = 1 / (2 * sigma_x ** 2 * varphi_x ** 2)
    term5 = 1 / (2 * sigma_y ** 2 * varphi_y ** 2)
    exponent = term1 - term2 - term3 - term4 - term5
    A_mod = A_0 * np.exp(exponent)
    return A_mod


def fading_loss():
    eta_t = transmissivity_etat(tau_zen, theta_zen_rad)
    eta_b = transmissivity_etab(a, r, waist)
    eta = eta_t * eta_b
    A_mod = mod_jitter()
    mu = rytov**2/2 * (1+2*varphi_mod**2)
    term1 = (varphi_mod**2) / (2 * (A_mod * eta_t)**(varphi_mod**2))
    term2 = eta**(varphi_mod**2) ** -1
    term3 = erfc((np.log((eta / (A_mod * eta_t)) + mu)) / (np.sqrt(2) * sigma_R))
    term4 = np.exp((sigma_R**2) / 2 * varphi_mod**2 * (1 + varphi_mod**2))
    
    f_eta = term1 * term2 * term3 * term4
    return f_eta