from scipy.special import lambertw
import math
import numpy as np


#=======================================================#
# Transmissivity Eta


#=======================================================#


#=======================================================#
# Scale function R
r = 

#=======================================================#


#=======================================================#
# Shape function lambda
def shape_lambda(xi, radius):
    pow_a = pow(radius, 2)
    pow_xi = pow(xi, 2)


#=======================================================#

#=======================================================#
# Intensity of elliptic beam
def intensity(w_1, w_2, phi, r, r0):
    #======================#
    # w_1: long axis of the elliptic
    # w_2: short axis of the elliptic
    # phi: angle of rotation of long axis w_1 with respect to the x-axis
    # r: position vector
    # r0: beam centre position vector
    #======================#

    # beam_centroid position
    pow_w1 = pow(w_1, 2)
    pow_w2 = pow(w_2, 2)
    s_xx = pow_w1 * pow(math.cos(phi), 2) + pow_w2 * pow(math.sin(phi), 2)
    s_yy = pow_w1 * pow(math.sin(phi), 2) + pow_w2 * pow(math.cos(phi), 2)
    s_xy = 1/2*(pow_w1 - pow_w2)*(2*math.sin(phi)*math.cos(phi))
    s = np.array([[s_xx, s_xy], [s_xy, s_yy]])

    # Compute the matrix S^-1
    s_inv = np.linalg.inv(s)

    # compute determinant det(S)
    det_s = np.linalg.det(s)

    # difference vector
    diff = r - r0

    # I(r, z)
    exponent = -2 * np.dot(np.dot(diff.T, s_inv), diff)
    intensity = (2 / np.pi) * np.sqrt(det_s) * np.exp(exponent)

    return intensity

#=======================================================#


#=======================================================#
# I_1
#=======================================================#

#=======================================================#
# Slot radius W_eff
## aperture of radius (m)
a = 0.35

## long axis
w_1 = 0.2*a

## short axis 
w_2 = 0.1*a

## Beam rotation angle (in radians) relative to the centroid axis
chi = math.pi/3

## exp and power exponent for slot radius function of exp
exp_power = math.exp((pow(a, 2) / pow(w_1, 2)) * (1 + 2 * pow(math.cos(math.pi / 5), 2)))

## beam_efficiency_factor
beam_efficiency_factor = 4*pow(a, 2) / (w_1 * w_2)

## argument for lambert w function
lambert_arg = pow(beam_efficiency_factor, exp_power) * exp_power

## lambert w function
lambert_val = lambertw(lambert_arg)

## slot radius W_eff
w_eff = math.sqrt(pow(4*pow(a, 2) * lambert_val, -1))
#=======================================================#