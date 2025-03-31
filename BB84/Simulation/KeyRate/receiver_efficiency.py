from scipy.special import lambertw
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
w_2 = 0.1 * a
r0 = 0.2
r1 = 0.25
phi = 3/math.pi
varphi0 = 4/math.pi
varphi1 = 5/math.pi
#=======================================================#



#=======================================================#
# Transmissivity Eta
#=======================================================#


#=======================================================#


#=======================================================#
# Scale function R
# Shape function lambda
#=======================================================#
def r_scale(xi):
    a2_xi2 = (a**2) * (xi**2)

    denominator = 1 - np.exp(-a2_xi2) * intensity_0()

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
    
    numerator = 2 * a2_xi2 * np.exp(-a2_xi2) * intensity_1()

    denominator = 1 - np.exp(-a2_xi2) * intensity_0()

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
def W_eff(w_1, w_2, phi, varphi):
    chi = phi - varphi
    exp_power = math.exp((pow(a, 2) / pow(w_1, 2)) * (1 + 2 * pow(math.cos(math.pi / 5), 2)))
    
    ## beam_efficiency_factor
    beam_efficiency_factor = 4*pow(a, 2) / (w_1 * w_2)

    ## argument for lambert w function
    lambert_arg = pow(beam_efficiency_factor, exp_power) * exp_power

    ## lambert w function
    lambert_val = lambertw(lambert_arg)

    ## slot radius W_eff
    w_eff = math.sqrt(pow(4*pow(a, 2) * lambert_val, -1))

    return w_eff
#=======================================================#
