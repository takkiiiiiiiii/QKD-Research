from scipy.special import lambertw, i0, i1
import math
import numpy as np


#=======================================================#
# Channel loss parameter
#=======================================================#
    #=====================#
    # D_r    : Deceiver diameter in meters
    # a      : Aperture of radius (Receiver radis in meters)
    # w_1    : Long axis of the elliptic
    # w_2    : Short axis of the elliptic
    # r0     : Distance of elliptical beam center from aperture center (開口中心から楕円ビームの中心の距離)
    # r1     : Distance of a point in the elliptical beam from the aperture center (開口中心から楕円ビーム内のある点の距離)
    # phi    : Angle between the x-axis and the elliptic semi-axis related to w_1^2
    # varphi0: Angle in polar coordinates, representing the position of the center point in the beam profile.
    # chi    : phi - varphi0
    # varphi1: Angle in polar coordinates, representing the position of a point in the beam profile
    #======================#
D_r = 0.35
a = D_r / 2
#=======================================================#


#=======================================================#
# Transmissivity nb(eta_b)
#=======================================================#
def transmissivity(beam_centroid_displacement, chi, w_1, w_2):
    eta_0 = transmissivity_0(w_1, w_2)
    exp_term = np.exp(-(beam_centroid_displacement/r_scale(2/W_eff(chi, w_1, w_2)))**lambda_shape(2/W_eff(chi, w_1, w_2)))
    eta = eta_0 * exp_term
    return eta
#=======================================================#


#=======================================================#
# Transmissivity n0(eta_0) : the transmittance for the
# centered beam
#=======================================================#
def transmissivity_0(w_1, w_2):
    # 変数の定義
    inv_W1_sq = 1 / w_1**2
    # print(inv_W1_sq)
    inv_W2_sq = 1 / w_2**2
    # print(inv_W2_sq)
    # print(inv_W1_sq-inv_W2_sq)
    delta_W = 1 / w_1 - 1 / w_2  # (1/W1 - 1/W2)
    
    # Bessel function I_0
    A = a**2 * (inv_W2_sq - inv_W1_sq)
    exp_term1 = np.exp(-a**2 * (inv_W1_sq + inv_W2_sq))
    I0_term = i0(A)  # scipy.special.i0(A) を使う

    I0_term = I0_term * exp_term1

    # Exponential term
    exp_term2 = np.exp(-0.5 * a**2 * delta_W)

    
    # Scaling function R and lambda
    R_xi = r_scale(delta_W) 
    lambda_xi = lambda_shape(delta_W)  
    
    # Second exponential term
    exp_term3 = np.exp(-((w_1 + w_2)**2 / np.abs(w_1**2 - w_2**2) / R_xi) ** lambda_xi)

    # Eta_0 の計算
    eta_0_val = 1 - I0_term - 2 * (1 - exp_term2) * exp_term3

    return eta_0_val

#=======================================================#


#=======================================================#
# Scale function R
# Shape function lambda
#=======================================================#
def safe_i0(x):
    return np.where(x > 100, np.exp(x) / np.sqrt(2 * np.pi * x), i0(x))


def r_scale(xi):
    a2_xi2 = (a**2) * (xi**2)

    if np.isnan(a2_xi2) or np.isinf(a2_xi2):
        raise ValueError(f"Error: a2_xi2 is invalid! a2_xi2 = {a2_xi2}, xi = {xi}")

    denominator = 1 - np.exp(-a2_xi2) * i0(a2_xi2)

    log_term = np.log(2 * (1 - np.exp(-0.5 * a2_xi2)) / denominator)

    lambda_pow = 1 / lambda_shape(xi)

    scale_xi = 1 / (pow(log_term, lambda_pow))

    return scale_xi
#=======================================================#


#=======================================================#
# Shape function lambda λ(ξ)
#=======================================================#
def safe_i1(x):
    return np.where(x > 100, np.exp(x) / np.sqrt(2 * np.pi * x), i1(x))  


def lambda_shape(xi):
    a2_xi2 = (a**2) * (xi**2)
    
    if np.isnan(a2_xi2) or np.isinf(a2_xi2):
        raise ValueError(f"Error: a2_xi2 is invalid! a2_xi2 = {a2_xi2}, xi = {xi}")
    
    numerator = 2 * a2_xi2 * np.exp(-a2_xi2) * i1(a2_xi2)

    denominator = 1 - np.exp(-a2_xi2) * i0(a2_xi2)

    log_term = np.log(2 * (1 - np.exp(-0.5 * a2_xi2)) / denominator)

    shape_xi = (numerator / denominator) * (1 / log_term)

    return shape_xi
#=======================================================#


#=======================================================#
# Slot radius W_err
#=======================================================#
def W_eff(chi, w_1, w_2):
    exp_power_cos = math.exp((pow(a, 2) / pow(w_1, 2)) * (1 + 2 * pow(math.cos(chi), 2)))
    exp_power_sin = math.exp((pow(a, 2) / pow(w_2, 2)) * (1 + 2 * pow(math.sin(chi), 2)))
    
    # beam_efficiency_factor の計算
    beam_efficiency_factor = 4 * pow(a, 2) / (w_1 * w_2)

    # lambert w 関数の引数
    lambert_arg = beam_efficiency_factor * exp_power_cos * exp_power_sin

    # lambert w の計算
    lambert_val = lambertw(lambert_arg).real  # 実数部分のみ取得

    # slot radius W_eff
    w_eff2 = 4 * pow(a, 2) * math.sqrt(1 / lambert_val)
    
    w_eff = math.sqrt(w_eff2)
    return w_eff
#=======================================================#

def to_decimal_string(x, precision=70):
    if x == 0:
        return "0." + "0" * precision
    return format(x, f'.{precision}f').rstrip('0').rstrip('.')


def simulation_eta_b():
    # =======Definition of parameter =========== #
    ratios = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    r0 = [r * a for r in ratios]
    mag_w1 = [0.2, 1.0, 1.8]
    mag_w2 = [0.1, 0.9, 1.7]
    chi = [math.pi/3, math.pi/4, math.pi/5]
    chi_show = [3, 4, 5]
    print("===============================")
    print(f'Aperture of radius (Receiver radis in meters): {a} m')
    for i, w_1 in enumerate(mag_w1):
        print(f'Long axis: {mag_w1[i]} * {a}')
        print(f'Long axis: {mag_w2[i]} * {a}')
        print(f'Chi: π / {chi_show[i]}')
    # print(f'Long axis of the elliptic: {w_1*a}')
    # print(f'Long axis of the elliptic: {w_2*a}')
        beam_centroid_displacement = [r / a for r in r0]
        eta_b = [transmissivity(b, chi[i], mag_w1[i]*a, mag_w2[i]*a) for b in beam_centroid_displacement]
        print("Transmissivity values:")
        for j, eta in enumerate(eta_b):
            print(f"  r0/a = {ratios[j]} → <ηb> = {to_decimal_string(eta)}")
        print("===============================\n")
    
    print("Simulation Finish !!")


def main():
    simulation_eta_b()

if __name__ == '__main__':
    main()