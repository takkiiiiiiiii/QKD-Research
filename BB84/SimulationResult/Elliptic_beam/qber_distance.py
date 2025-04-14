from scipy.special import lambertw, i0, i1
import math
import numpy as np
import matplotlib.pyplot as plt
import os, sys
simulation_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Model'))
sys.path.append(simulation_path)
from beam_transmissivity import transmissivity_etab
from atmospheric_transmissivity import transmissivity_etat, to_decimal_string


D_r = 0.35 # D_r    : Deceiver diameter in meters
a = D_r/2  # a      : Aperture of radius (Receiver radis in meters)
ratios = np.arange(0, 3.1, 0.1)
r0 = [r * a for r in ratios]
mag_w1 = [0.2, 1.0, 1.1]
mag_w2 = [0.1, 0.9, 1.0]
chi = [math.pi/3, math.pi/4, math.pi/5]
chi_show = [3, 4, 5]


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

    print("===============================")
    print(f'Aperture of radius (Receiver radis in meters): {a} m')

    # print(f'Long axis: {mag_w1[i]} * {a}')
    # print(f'Long axis: {mag_w2[i]} * {a}')
    # print(f'Chi: π / {chi_show[i]}')
    # beam_centroid_displacement = [r / a for r in r0]
    beam_centroid_displacement = 0
    all_eta_b = [transmissivity_etab(beam_centroid_displacement, chi[i], mag_w1[i] * a, mag_w2[i] * a) for i in range(len(chi))]
    all_eta_t = [transmissivity_etat(d_b) for d_b in distances]
    # gamma_list = [eta_b * eta_t for _, (eta_t) in enumerate(zip(all_eta_t))]
    for i, eta_b in enumerate(all_eta_b):
        gamma_list = []
        print(f'Long axis: {mag_w1[i]} * {a}')
        print(f'Long axis: {mag_w2[i]} * {a}')
        print(f'Chi: π / {chi_show[i]}')
        for j, eta_t in enumerate(all_eta_t):
            gamma = eta_b * eta_t
            gamma_list.append(gamma)
            print(f'Distance = {distances[j]}m, 𝛾 (eta_b × eta_t) = {to_decimal_string(gamma)}')  # ここで出力

        all_gamma.append(gamma_list)

        # print("Transmissivity values:")
        # for j, gamma in enumerate(gamma_list):
        #     print(f"  r0/a = {ratios[j]} → <ηb> = {to_decimal_string(gamma)}")
        
        # all_gamma.append(gamma_list)  # 各`eta_b`リストを追加
    
    print("===============================\n")
    print("Simulation Finish !!")
    return all_gamma  # 全てのeta_bリストを返す

def main():
    all_prob_error = []
    all_gamma = simulation_gamma()
    for i, _ in enumerate(zip(mag_w1, mag_w2, chi)):
        prob_error_result = []
        print(f'Chi: π / {chi_show[i]}')
        for j, _ in enumerate(distances):
            gamma = all_gamma[i][j]
            # print(f'𝛾 = {to_decimal_string(gamma)}')
            prob_error = qber(gamma)
            print(f'BER = {to_decimal_string(prob_error)}')
            prob_error_result.append(prob_error)
        all_prob_error.append(prob_error_result)

    plt.figure(figsize=(10, 6))
    for i, prob_error in enumerate(all_prob_error):
        # plt.plot(ratios, ber_results, marker='o', label=f'Condition {i+1}')
        plt.plot(distances, prob_error, marker='o', label=f'χ = π/{chi_show[i]}')

    # グラフの描画
    # plt.figure()
    plt.xlabel('Distance(m)')
    plt.ylabel('BER (%)')
    plt.title(f'BER vs Distance')
    plt.legend()
    plt.grid(True)
    output_path = os.path.join(os.path.dirname(__file__), "qber_distance_plot.png")
    plt.savefig(output_path)
    print(f"✅ Save as: {output_path}")
    plt.show()   


if __name__ == "__main__":
    main()


0.00000003595331600342313729413838176152129477713970118202269077301025390625
0.00000003595331600342313729413838176152129477713970118202269077301025390625