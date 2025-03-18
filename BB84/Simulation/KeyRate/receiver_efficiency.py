from scipy.special import lambertw
import math


#=======================================================#
# Transmissivity Eta


#=======================================================#


#=======================================================#
# Scale function R


#=======================================================#

#=======================================================#
# Shape function A



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