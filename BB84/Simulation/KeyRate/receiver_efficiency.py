from scipy.special import lambertw, i0, i1
from scipy.special import *
import math
import numpy as np


#=======================================================#
# Channel loss parameter
#=======================================================#
    #=====================#
    # a      : aperture of radius
    # w_1    : long axis of the elliptic
    # w_2    : short axis of the elliptic
    # r0     : Distance of elliptical beam center from aperture center (開口中心から楕円ビームの中心の距離)
    # r1     : Distance of a point in the elliptical beam from the aperture center (開口中心から楕円ビーム内のある点の距離)
    # phi    : angle between the x-axis and the elliptic semi-axis related to w_1^2
    # varphi0: angle in polar coordinates, representing the position of the center point in the beam profile.
    # varphi1: angle in polar coordinates, representing the position of a point in the beam profile
    #======================#

a = 0.35
w_1 = 0.2 * a
w_2 = 0.3 * a
r0 = 0.2
r1 = 0.25
phi = 3/math.pi
varphi0 = 4/math.pi
varphi1 = 5/math.pi
#=======================================================#



#=======================================================#
# Transmissivity nb(eta_b)
#=======================================================#
def transmissivity():
    eta_0 = transmissivity_0()
    exp_term = np.exp(-(r0/a/r_scale(2/W_eff(phi, varphi0)))**lambda_shape(2/W_eff(phi, varphi0)))
    eta = eta_0 * exp_term
    return eta


#=======================================================#


#=======================================================#
# Transmissivity n0(eta_0) : the transmittance for the
# centered beam
#=======================================================#
def transmissivity_0():
    # 変数の定義
    inv_W1_sq = 1 / w_1**2
    print(inv_W1_sq)
    inv_W2_sq = 1 / w_2**2
    print(inv_W2_sq)
    print(inv_W1_sq-inv_W2_sq)
    delta_W = 1 / w_1 - 1 / w_2  # (1/W1 - 1/W2)
    
    # Bessel function I_0
    A = a**2 * (inv_W1_sq - inv_W2_sq) * np.exp(-a**2 * (inv_W1_sq + inv_W2_sq))
    I0_term = i0(A)  # scipy.special.i0(A) を使う

    # Exponential term
    exp_term1 = np.exp(-0.5 * a**2 * (inv_W1_sq - inv_W2_sq)**2)
    
    # Scaling function R and lambda
    R_xi = r_scale(delta_W)  # `r_scale(xi)` を定義済みとする
    lambda_xi = lambda_shape(delta_W)  # `lambda_shape(xi)` を定義済みとする
    
    # Second exponential term
    exp_term2 = np.exp(-((w_1 + w_2)**2 / np.abs(w_1**2 - w_2**2) / r_scale(inv_W1_sq-inv_W2_sq)) ** lambda_shape(inv_W1_sq-inv_W2_sq))

    # Eta_0 の計算
    eta_0_val = 1 - I0_term - 2 * (1 - exp_term1) * exp_term2

    return eta_0_val

#=======================================================#


#=======================================================#
# Scale function R
# Shape function lambda
#=======================================================#
def r_scale(xi):
    a2_xi2 = (a**2) * (xi**2)

    # デバッグ用の出力
    print(f"DEBUG: a2_xi2 = {a2_xi2}, xi = {xi}")

    # NaN や Inf をチェック
    if np.isnan(a2_xi2) or np.isinf(a2_xi2):
        raise ValueError(f"Error: a2_xi2 is invalid! a2_xi2 = {a2_xi2}, xi = {xi}")

    denominator = 1 - np.exp(-a2_xi2) * i0(a2_xi2)

    log_term = np.log(2 * (1 - np.exp(-0.5 * a2_xi2)) / denominator)

    # WIP:modify

    lambda_pow = 1 / lambda_shape(xi)

    scale_xi = 1 / (pow(log_term, lambda_pow))

    return scale_xi
#=======================================================#


#=======================================================#
# Shape function lambda λ(ξ)
#=======================================================#
def lambda_shape(xi):
    a2_xi2 = (a**2) * (xi**2)
    
    # デバッグ用の出力
    print(f"DEBUG: a2_xi2 = {a2_xi2}, xi = {xi}")

    # NaN や Inf をチェック
    if np.isnan(a2_xi2) or np.isinf(a2_xi2):
        raise ValueError(f"Error: a2_xi2 is invalid! a2_xi2 = {a2_xi2}, xi = {xi}")
    
    numerator = 2 * a2_xi2 * np.exp(-a2_xi2) * i1(a2_xi2)

    denominator = 1 - np.exp(-a2_xi2) * i0(a2_xi2)

    log_term = np.log(2 * (1 - np.exp(-0.5 * a2_xi2)) / denominator)

    shape_xi = (numerator / denominator) * (1 / log_term)

    return shape_xi
#=======================================================#



#=======================================================#
# Intensity of elliptic beam defined I_0 (r = r0)
#=======================================================#
def intensity_0():
    # beam_centroid position
    pow_w1 = pow(w_1, 2)
    pow_w2 = pow(w_2, 2)
    s_xx = pow_w1 * pow(math.cos(phi), 2) + pow_w2 * pow(math.sin(phi), 2)
    s_yy = pow_w1 * pow(math.sin(phi), 2) + pow_w2 * pow(math.cos(phi), 2)
    s_xy = 1/2*(pow_w1 - pow_w2)*(2*math.sin(phi)*math.cos(phi))
    s = np.array([[s_xx, s_xy], [s_xy, s_yy]])

    # compute determinant det(S)
    det_s = np.linalg.det(s)

    # I(r, z)
    intensity = (2 / np.pi) * np.sqrt(det_s) 
    return intensity
#=======================================================#


#=======================================================#
# Intensity of elliptic beam which defined I_1
#=======================================================#
def intensity_1():
    vec_r1 = np.array([r1*math.cos(varphi1), r1*math.sin(varphi1)])  
    vec_r0 = np.array([r0*math.cos(varphi0), r0*math.sin(varphi0)])

    # beam_centroid position
    pow_w1 = pow(w_1, 2)
    pow_w2 = pow(w_2, 2)
    s_xx = pow_w1 * pow(math.cos(phi), 2) + pow_w2 * pow(math.sin(phi), 2)
    s_yy = pow_w1 * pow(math.sin(phi), 2) + pow_w2 * pow(math.cos(phi), 2)
    s_xy = 1/2*(pow_w1 - pow_w2)*(2*math.sin(phi)*math.cos(phi))
    s = np.array([[s_xx, s_xy], [s_xy, s_yy]])

    # Compute the matrix S^-1
    s_inv = np.linalg.inv(s)

    # difference vector
    diff = vec_r0 - vec_r1

    # I(r, z)
    # I_0: The intensity is the maximum value of the intensity at the centre of the beam (i.e. at position r=r0).
    i_0 = intensity_0(w_1, w_2, phi)
    exponent = -2 * np.dot(np.dot(diff.T, s_inv), diff)
    intensity = i_0 * np.exp(exponent)

    return intensity
#=======================================================#


#=======================================================#
# Slot radius W_err
#=======================================================#
def W_eff(phi, varphi):
    chi = phi - varphi
    exp_power = math.exp((pow(a, 2) / pow(w_1, 2)) * (1 + 2 * pow(math.cos(chi), 2)))
    
    # beam_efficiency_factor の計算
    beam_efficiency_factor = 4 * pow(a, 2) / (w_1 * w_2)

    # lambert w 関数の引数
    lambert_arg = beam_efficiency_factor * exp_power * exp_power  # 修正: スカラー値にする

    # lambert w の計算
    lambert_val = lambertw(lambert_arg).real  # 実数部分のみ取得

    # slot radius W_eff
    w_eff = math.sqrt(1 / (4 * pow(a, 2) * lambert_val))  # 修正: pow の位置

    print(f'Efficiency: {w_eff}')
    return w_eff
#=======================================================#

def main():
    eta_b = transmissivity()
    print(f'Transmissivity: {eta_b}')


if __name__ == '__main__':
    main()