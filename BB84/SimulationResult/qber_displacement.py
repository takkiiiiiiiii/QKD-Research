from scipy.special import lambertw, i0, i1
import math
import numpy as np
import matplotlib.pyplot as plt
import os, sys
simulation_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Model'))
sys.path.append(simulation_path)
from receiver_efficiency import transmissivity, to_decimal_string

D_r = 0.35 # D_r    : Deceiver diameter in meters
a = D_r/2  # a      : Aperture of radius (Receiver radis in meters)
ratios = np.arange(0, 3.1, 0.1)
r0 = [r * a for r in ratios]
mag_w1 = [0.2, 1.0, 1.8]
mag_w2 = [0.1, 0.9, 1.7]
chi = [math.pi/3, math.pi/4, math.pi/5]
chi_show = [3, 4, 5]


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

    for i, w_1 in enumerate(mag_w1):
        print(f'Long axis: {mag_w1[i]} * {a}')
        print(f'Long axis: {mag_w2[i]} * {a}')
        print(f'Chi: π / {chi_show[i]}')

        beam_centroid_displacement = [r / a for r in r0]
        gamma = [transmissivity(b, chi[i], mag_w1[i] * a, mag_w2[i] * a) for b in beam_centroid_displacement]
        
        print("Transmissivity values:")
        for j, eta in enumerate(gamma):
            print(f"  r0/a = {ratios[j]} → <ηb> = {to_decimal_string(eta)}")
        
        all_gamma.append(gamma)  # 各`eta_b`リストを追加
    
        print("===============================\n")
    
    print("Simulation Finish !!")
    return all_gamma  # 全てのeta_bリストを返す

def main():
    all_prob_error = []
    all_gamma = simulation_gamma()
    for i, (mag_w1_value, mag_w2_value, chi_value) in enumerate(zip(mag_w1, mag_w2, chi)):
        prob_error_result = []
        for j, _ in enumerate(ratios):
            gamma = all_gamma[i][j]
            prob_error = qber(gamma)
            prob_error_result.append(prob_error)
        all_prob_error.append(prob_error_result)

    plt.figure(figsize=(10, 6))
    for i, prob_error in enumerate(all_prob_error):
        # plt.plot(ratios, ber_results, marker='o', label=f'Condition {i+1}')
        plt.plot(ratios, prob_error, marker='o', label=f'χ = π/{chi_show[i]}')

    # グラフの描画
    # plt.figure()
    plt.xlabel('r0/a')
    plt.ylabel('BER (%)')
    plt.title(f'BER vs r0/a')
    plt.legend()
    plt.grid(True)
    output_path = os.path.join(os.path.dirname(__file__), "qber_displacement_plot.png")
    plt.savefig(output_path)
    print(f"✅ Save as: {output_path}")
    plt.show()   


if __name__ == "__main__":
    main()