# Upgrade qiskit version to 2.0.0
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (NoiseModel, pauli_error)
import math
import numpy as np
import socket
import os, sys
import matplotlib.pyplot as plt
simulation_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Model'))
sys.path.append(simulation_path)
from receiver_efficiency import transmissivity


backend = AerSimulator()

#===============Deefinition of parameter===============#
count = 1000
no_qubits = 20
D_r = 0.35 # D_r    : Deceiver diameter in meters
a = D_r/2  # a      : Aperture of radius (Receiver radis in meters)
ratios = np.arange(0, 3.1, 0.1)
r0 = [r * a for r in ratios]
mag_w1 = [0.2, 1.0, 1.8]
mag_w2 = [0.1, 0.9, 1.7]
chi = [math.pi/3, math.pi/4, math.pi/5]
chi_show = [3, 4, 5]


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


def generate_Siftedkey(user0, user1, num_qubits, eta_b):
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
    noise_model = apply_noise_model(eta_b)

    # Bob measures Alice's qubit
    qc, bob_bits = bob_measurement(qc, bob_basis, noise_model)

    altered_qubits = 0

    user0.create_socket_for_classical()
    user1.create_socket_for_classical()
    sender_classical_channel = user0.socket_classical
    receiver_classical_channel = user1.socket_classical

    ka = ''  # Alice's sifted key
    kb = ''  # Bob's sifted key
    err_num = 0

    # Announce bob's basis
    receiver_classical_channel.send(bob_basis.encode('utf-8'))
    bob_basis = sender_classical_channel.recv(4096).decode('utf-8')
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


    sender_classical_channel.close()
    receiver_classical_channel.close()

    return ka, kb


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


def apply_noise_model(eta_b):
    error_meas = pauli_error([('X', 1-eta_b), ('I', eta_b)])
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


# Transmissity
def simulation_eta_b():
    # =======Definition of parameter =========== #
    all_eta_b = []

    print("===============================")
    print(f'Aperture of radius (Receiver radis in meters): {a} m')

    for i, w_1 in enumerate(mag_w1):
        print(f'Long axis: {mag_w1[i]} * {a}')
        print(f'Long axis: {mag_w2[i]} * {a}')
        print(f'Chi: π / {chi_show[i]}')

        beam_centroid_displacement = [r / a for r in r0]
        eta_b = [transmissivity(b, chi[i], mag_w1[i] * a, mag_w2[i] * a) for b in beam_centroid_displacement]
        
        # print("Transmissivity values:")
        # for j, eta in enumerate(eta_b):
        #     print(f"  r0/a = {ratios[j]} → <ηb> = {to_decimal_string(eta)}")
        
        all_eta_b.append(eta_b)  # 各`eta_b`リストを追加
    
        print("===============================\n")
    
    print("Simulation Finish !!")
    return all_eta_b  # 全てのeta_bリストを返す


# Calculate the bit error rate
def calculate_ber(ka, kb):
    # キーの長さが一致していることを確認
    if len(ka) != len(kb):
        raise ValueError("Keys must be of the same length")

    # エラー数のカウント
    error_count = sum(1 for bit_a, bit_b in zip(ka, kb) if bit_a != bit_b)

    # BERの計算
    ber = (error_count / len(ka)) * 100
    return ber


def main():
    all_ber_results = []
    all_eta_b = simulation_eta_b()
    for i, (mag_w1_value, mag_w2_value, chi_value) in enumerate(zip(mag_w1, mag_w2, chi)):
        ber_results = []
        for j, ratio in enumerate(ratios):
            eta_b = all_eta_b[i][j]
            ka = ''
            kb = ''
            for _ in range(count):
                part_ka, part_kb = generate_Siftedkey(user0, user1, no_qubits, eta_b)
                ka += part_ka
                kb += part_kb
            ber = calculate_ber(ka, kb)
            ber_results.append(ber)
        all_ber_results.append(ber_results)

    plt.figure(figsize=(10, 6))
    for i, ber_results in enumerate(all_ber_results):
        # plt.plot(ratios, ber_results, marker='o', label=f'Condition {i+1}')
        plt.plot(ratios, ber_results, marker='o', label=f'χ = π/{chi_show[i]}')

    # グラフの描画
    # plt.figure()
    plt.xlabel('r0/a')
    plt.ylabel('BER (%)')
    plt.title(f'BER vs r0/a')
    plt.legend()
    plt.grid(True)
    output_path = os.path.join(os.path.dirname(__file__), "qber_plot.png")
    plt.savefig(output_path)
    print(f"✅ Save as: {output_path}")
    plt.show()


if __name__ == '__main__':
    main()
