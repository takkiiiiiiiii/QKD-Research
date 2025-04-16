from scipy.special import lambertw, i0, i1
import math
import numpy as np
import matplotlib.pyplot as plt
import os, sys
simulation_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'Model'))
sys.path.append(simulation_path)
from circle_beam_transmissivity import transmissivity_etab
from atmospheric_transmissivity import transmissivity_etat, to_decimal_string


D_r = 0.35 # D_r    : Deceiver diameter in meters
a = D_r/2  # a      : Aperture of radius (Receiver radis in meters)
W = [0.2*a, 1.0*a, 1.3*a]
# W_show = [0.2, 1.0, 1.8]
# W_show = [w*a for w in W]
W_show = [w / a for w in W]


#=======================================================#
# atmospheric transmissivity(eta_t) parameter
#=======================================================#
    #=====================#
    # distance    : the light-of-sight(LOS) distance(m) between Alice and Bob
    #=====================#
#=======================================================#
# distances = np.arange(100000, 2100000, 100000)
distances = np.arange(100, 1600, 100) 


#=======================================================#
# Channel loss parameter
#=======================================================#
    #=====================#
    # n_s   : average numher of photon
    # n_D   : dark-current-equivalent averrage photon number
    # eta   : 
    # n_B   : background photons per polarization
    # n_N   : the average number of noise photon reaching each detector
    # gamma : the fraction of transmmited photon
    #======================#
n_s = 20
n_D = math.pow(10, -6)
eta = 0.5  
n_B = math.pow(10, -3)
n_N = n_B/2 + n_D



def qber(gamma):
    # prob_error = eta * n_N * math.exp(-eta(n_s*gamma+4*n_N))
    prob_error = eta * n_N * math.exp(-eta * (n_s * gamma + 4 * n_N))
    return prob_error


def simulation_gamma():
    # =======Definition of parameter =========== #
    all_gamma = []


    # beam_centroid_displacement = [r / a for r in r0]
    # displacement = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    # r = [a*d for d in displacement]
    displacement = 0.5
    r = a*displacement
    all_eta_b = [transmissivity_etab(a, r, w) for w in W]
    all_eta_t = [transmissivity_etat(d_b) for d_b in distances]
    # gamma_list = [eta_b * eta_t for _, (eta_t) in enumerate(zip(all_eta_t))]
    for i, eta_b in enumerate(all_eta_b):
        gamma_list = []
        for j, eta_t in enumerate(all_eta_t):
            gamma = eta_b * eta_t
            gamma_list.append(gamma)

        all_gamma.append(gamma_list)

        # for j, gamma in enumerate(gamma_list):
        
        # all_gamma.append(gamma_list)
    
    return all_gamma  

def main():
    all_prob_error = []
    all_gamma = simulation_gamma()
    for i, _ in enumerate(W):
        prob_error_result = []
        for j, _ in enumerate(distances):
            gamma = all_gamma[i][j]
            prob_error = qber(gamma)
            prob_error_result.append(prob_error)
        all_prob_error.append(prob_error_result)

    plt.figure(figsize=(10, 6))
    for i, prob_error in enumerate(all_prob_error):
        # plt.plot(ratios, ber_results, marker='o', label=f'Condition {i+1}')
        plt.plot(distances, prob_error, marker='o', label=f'w={W_show[i]}a')

    # グラフの描画
    # plt.figure()
    plt.xlabel('Distance(m)')
    plt.ylabel('BER (%)')
    plt.title(f'BER vs Distance')
    plt.legend()
    plt.grid(True)
    output_path = os.path.join(os.path.dirname(__file__), "qber_distance_plot.png")
    plt.savefig(output_path)
    plt.show()   


if __name__ == "__main__":
    main()