# Upgrade qiskit version to 2.0.0
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (NoiseModel, pauli_error)
import time
import socket, math
from ave_qber_zenith import *
import numpy as np
import os, sys
import matplotlib.pyplot as plt
from kr_Hamming import key_reconciliation_Hamming
from finalkeyrate import compute_final_keyrate



backend = AerSimulator()
intercept_prob = 0

#=======================================================#
#                 Fading Parameters                     #
#=======================================================#
#==================================================================#
# a : Aperture of radius (Receiver radis in meters) (m)
#==================================================================#
a = 0.75

# r = 5
#==================================================================#
# n_s : average number of photon
#==================================================================#
n_s = 0.1

#==================================================================#
# len_wave : Optical wavelength (μm)
#==================================================================#
lambda_ = 0.85e-6

#==================================================================#
# altitude ground station
#==================================================================#
H_g = 10 # (m)

#==================================================================#
# h_s : Altitude between LEO satellite and ground station (m)
#==================================================================#
h_s = 500e3  # 500 km

#==================================================================#
# H_a : Upper end of atmosphere (km)
#==================================================================#
H_atm = 200000

#==================================================================#
# theta_d_rad : Optical beam divergence angle (rad)
#==================================================================#
theta_d_rad = 10e-6 

#==================================================================#
# theta_d_half_rad : Optical beam half-divergence angle (rad)
#==================================================================#
theta_d_half_rad = theta_d_rad /2

#==================================================================#
# v_wind: wind_speed
#==================================================================#
v_wind = 21 

#==================================================================#
# mu_x, mu_y: Mean values of pointing error in x and y directions (m)
#==================================================================#
mu_x = 0
mu_y = 0

#==================================================================#
# angle_sigma_x, angle_sigma_y: Beam jitter standard deviations of the Gaussian-distibution jitters (rad)
#==================================================================#
angle_sigma_x = 3e-6
angle_sigma_y = 3e-6

#=======================================================#
# QBER parameters
#=======================================================#
    #=====================#
    # n_s   : average numher of photon from Alice
    # e_0   : the error rate of the background
    # Y_0   : the background rate which includes the detector dark count and other background contributions
    # P_pa  : After-pulsing probability
    # e_pol : Probability of the polarisation errors
    #======================#i
e_0 = 0.5
Y_0 = 1e-4
P_AP = 0.02
e_pol = 0.033

#=======================================================#
# Final key settings
#=======================================================#
    #=====================#
    # s: sifting efficiency
    # p: Parameter estimation coefficient
    # f: Key reconciliation efficiency 
    #======================#i
s = 0.5
p = 0.75
f = 1.22


class User:
    def __init__(self, username: str, sharekey, socket_classical, socket_quantum):
        self.username = username
        self.sharekey = sharekey
        self.socket_classical = socket_classical
        self.socket_quantum = socket_quantum

    def create_socket_for_classical(self):
        SERVER_HOST_CLASSICAL = '127.0.0.1'
        SERVER_PORT_CLASSICAL = 12001
        client_socket_classical = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket_classical.connect((SERVER_HOST_CLASSICAL, SERVER_PORT_CLASSICAL))
        self.socket_classical = client_socket_classical        

user0 = User("Alice", None, None, None) 
user1 = User("Bob", None, None, None)


def generate_Siftedkey(user0, user1, num_qubits, ave_qber):
    start = time.time()
    alice_bits = qrng(num_qubits)
    alice_basis = qrng(num_qubits)
    bob_basis = qrng(num_qubits)
    eve_basis = qrng(num_qubits)

    # Alice generates qubits
    qc = compose_quantum_circuit(num_qubits, alice_bits, alice_basis)

    # Quantum Circuit for Eve
    qc2 = compose_quantum_circuit_for_eve(num_qubits, alice_bits, alice_basis)

    # Comparison their basis between Alice and Eve
    ae_basis, ae_match = check_bases(alice_basis, eve_basis)

    # Apply the quantum error chanel
    noise_model = apply_noise_model(ave_qber)

    # Bob measures Alice's qubit
    qc, bob_bits = bob_measurement(qc, bob_basis, noise_model)

    altered_qubits = 0

    # user0.create_socket_for_classical()
    # user1.create_socket_for_classical()
    # sender_classical_channel = user0.socket_classical
    # receiver_classical_channel = user1.socket_classical

    ka = ''  # Alice's sifted key
    kb = ''  # Bob's sifted key
    err_num = 0

    # Announce bob's basis
    # receiver_classical_channel.send(bob_basis.encode('utf-8'))
    # bob_basis = sender_classical_channel.recv(4096).decode('utf-8')
    ab_basis, ab_matches = check_bases(alice_basis, bob_basis)
    ab_bits = check_bits(alice_bits, bob_bits, ab_basis)

    for i in range(num_qubits):
        if ae_basis[i] != 'Y' and ab_basis[i] == 'Y': # Alice and Eve bases differ
            altered_qubits += 1
        if ab_basis[i] == 'Y':  # Alice and Bob bases match
            ka += alice_bits[i]
            kb += bob_bits[i]
        if ab_bits[i] == '!':  # Bits differ
            err_num += 1
    err_str = ''.join(['!' if ka[i] != kb[i] else ' ' for i in range(len(ka))])

    end = time.time()
    runtime = end - start

    # sender_classical_channel.close()
    # receiver_classical_channel.close()

    return ka, kb, err_num, runtime


def qrng(n):
    qc = QuantumCircuit(n, n)
    for i in range(n):
        qc.h(i)  # Apply Hadamard gate
    qc.measure(range(n), range(n))
    job_result = backend.run(qc, shots=1).result()
    counts = job_result.get_counts()

    # 取得した測定結果の中で最も出現回数が多いものを採用
    max_key = max(counts, key=counts.get)
    bits = ''.join(reversed(max_key))  # ビット列を逆順にして取得

    return bits


def encode_qubits(n, k, a):
    qc = QuantumCircuit(n, n)
    for i in range(n):
        if a[i] == '0':
            if k[i] == '1':
                qc.x(i)
        else:
            if k[i] == '0':
                qc.h(i)
            else:
                qc.x(i)
                qc.h(i)
    qc.barrier()
    return qc


def compose_quantum_circuit(num_qubit, alice_bits, alice_basis) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubit, num_qubit)
    qc.compose(encode_qubits(num_qubit, alice_bits, alice_basis), inplace=True)
    return qc


def compose_quantum_circuit_for_eve(num_qubit, alice_bits, alice_basis) -> QuantumCircuit:
    qc2 = QuantumCircuit(num_qubit, num_qubit)
    qc2.compose(encode_qubits(num_qubit, alice_bits, alice_basis), inplace=True)
    return qc2


def apply_noise_model(ave_qber):
    error_meas = pauli_error([('X', ave_qber), ('I', 1 - ave_qber)])
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(error_meas, "measure")
    return noise_model

def bob_measurement(qc, bob_basis, noise_model):
    for i in range(len(bob_basis)):
        if bob_basis[i] == '1':  # Diagonal basis
            qc.h(i)

    qc.measure(range(len(bob_basis)), range(len(bob_basis)))

    # Qiskit 2.0.0 では execute() を使わず backend.run(qc) を直接使用
    # result = backend.run(qc, shots=1).result()
    result = backend.run(qc, shots=1, noise_model=noise_model).result()
    counts = result.get_counts()  # `.get_counts(0)` ではなく `.get_counts()` に変更

    # 取得した測定結果の中で最も出現回数が多いものを採用
    max_key = max(counts, key=counts.get)
    bits = ''.join(reversed(max_key))  # ビット列を逆順にして取得

    qc.barrier()
    return qc, bits


def check_bases(b1, b2):
    check = ''
    matches = 0
    for i in range(len(b1)):
        if b1[i] == b2[i]:
            check += "Y"
            matches += 1
        else:
            check += "-"
    return check, matches


def check_bits(b1, b2, bck):
    check = ''
    for i in range(len(b1)):
        if b1[i] == b2[i] and bck[i] == 'Y':
            check += 'Y'
        elif b1[i] == b2[i] and bck[i] != 'Y':
            check += 'R'
        elif b1[i] != b2[i] and bck[i] == 'Y':
            check += '!'
        elif b1[i] != b2[i] and bck[i] != 'Y':
            check += '-'
    return check


def calculate_pulse_rate(n_s, num_qubits):
    return num_qubits / n_s


def main():
    num_samples = 1
    raw_keyrate = 6383.91
    kr_efficiency = 1.22
    sifting_coefficient = 0.5
    p_estimation = 0.9
    tau_zen_list = [0.91, 0.85, 0.75, 0.53]
    theta_zen_deg_list = np.linspace(-50, 50, 100)
    total_qubit = int(5000)
    num_qubits = 240
    num_running = total_qubit/num_qubits +1
    pulse_rate = calculate_pulse_rate(n_s, num_qubits)
    print(f'Pulse Rate: {pulse_rate} (pulse/sec)')
    print(f'{n_s} (photon/pulse)')
    for tau_zen in tau_zen_list:
        final_keyrate_values = []

        for theta_zen_deg in theta_zen_deg_list:
            theta_zen_rad = math.radians(theta_zen_deg)
            theta_zen_rad = math.radians(theta_zen_deg)
            LoS = satellite_ground_distance(h_s, H_g, theta_zen_rad)
            r = compute_radial_displacement(mu_x, mu_y, angle_sigma_x, angle_sigma_y, LoS, size=1)
            w_L = compute_w_L(lambda_, theta_d_half_rad, LoS, H_atm, H_g, theta_zen_rad, Cn2_profile)

            final_keyrate_samples = []
            prob_error = qner_new_infinite(theta_zen_rad, H_atm, w_L, tau_zen, LoS)
            qber_samples = []
            for _ in range(num_samples):
                total_err_num = 0
                total_sifted_bit_length = 0
                # 一回のシフト鍵生成
                for _ in range(int(num_running)):

                    part_ka, part_kb, err_num, runtime = generate_Siftedkey(
                        user0, user1, num_qubits, prob_error
                    )
                    total_err_num += err_num
                    total_sifted_bit_length += len(part_ka) 
                
                qber = total_err_num / total_sifted_bit_length * 100 if len(part_ka) > 0 else 0
                    # raw_keyrate = len(part_ka) / runtime
                    # raw_keyrate_samples.append(raw_keyrate)
                final_keyrate = compute_final_keyrate(
                    raw_keyrate, qber, sifting_coefficient, p_estimation, kr_efficiency
                )
                final_keyrate_samples.append(final_keyrate)

            avg_final_keyrate = sum(final_keyrate_samples) / len(final_keyrate_samples)
            # avg_raw_keyrate = sum(raw_keyrate_samples) / len(raw_keyrate_samples)
            print(f"θ_zen [deg] = {theta_zen_deg:.2f}, Avg Raw Keyrate = {raw_keyrate:.2f} bps")

            final_keyrate_values.append(avg_final_keyrate)

        label = weather_condition(tau_zen) + f" (τ = {tau_zen})"
        plt.plot(theta_zen_deg_list, final_keyrate_values, label=label)

    plt.xlabel(r"Zenith angle $\theta_{\mathrm{zen}}$ [deg]", fontsize=20)
    plt.ylabel("Final Key Rate [bps]", fontsize=20)
    plt.legend(fontsize=12)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(True)
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(__file__), f'finalkeyrate_vs_zenith_all_conditions_{n_s}.png')
    plt.savefig(output_path)
    plt.show()
    

if __name__ == '__main__':
    main()


