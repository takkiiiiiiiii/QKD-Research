import math
import numpy as np

#The atmospheric transmissivity(defined as eta_t) consider absorption and scattering.

#=======================================================#
# atmospheric transmissivity(eta_t) parameter
#=======================================================#
    #=====================#
    # k(kappa): atmospheric absortion coefficient in dB/m
    # d_b     : the light-of-sight(LOS) distance(m) between Alice and Bob
    #=====================#
#=======================================================#



def transmissivity_etab(d_b):
    kappa = 0.43 * pow(10, -3)
    eta_t = pow(10, -0.1*kappa*d_b)
    return eta_t

def to_decimal_string(x, precision=150):
    if x == 0:
        return "0." + "0" * precision
    return format(x, f'.{precision}f').rstrip('0').rstrip('.')

def main():
    # distances = np.arange(100, 1600, 100) 
    # Starlink uses Low-Earth Orbit (LEO) satellites to orbit the Earth at a height of between 180 to 2,000km.
    distances = np.arange(100000, 2100000, 100000) 
    all_eta_t = [transmissivity_etab(d_b) for d_b in distances]
    for j, eta_t in enumerate(all_eta_t):
        print(f"d_b = {distances[j]} → <ηt> = {to_decimal_string(eta_t)}")
        print("===============================\n")
    print("Simulation Finish !!\n")



if __name__ == '__main__':
    main()