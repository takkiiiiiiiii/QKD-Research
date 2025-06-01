from qiskit import QuantumCircuit
from qiskit_aer.noise import (NoiseModel, pauli_error)

from qiskit_aer import AerSimulator
# from kr_Hamming import key_reconciliation_Hamming
from IPython.display import display
# from qiskit.tools.visualization import plot_histogram
import numpy as np
import random
import math
import time



backend = AerSimulator()



class User:
    def __init__(self, username: str, sharekey, socket_classical, socket_quantum):
        self.username = username
        self.sharekey = sharekey
        self.socket_classical = socket_classical
        self.socket_quantum = socket_quantum

    def create_socket_for_classical(self):
        import socket
        SERVER_HOST_CLASSICAL = '127.0.0.1'
        SERVER_PORT_CLASSICAL = 12001
        client_socket_classical = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket_classical.connect((SERVER_HOST_CLASSICAL, SERVER_PORT_CLASSICAL))
        self.socket_classical = client_socket_classical        

user0 = User("Alice", None, None, None) 
user1 = User("Bob", None, None, None)



def generate_Siftedkey(user0, user1, num_qubits, noise_prob, intercept_prob):
    num_bellState = int(num_qubits/2)
    alice_basis = qrng(num_bellState)
    bob_basis = qrng(num_bellState)
    eve_basis = qrng(num_bellState)

    # Compose the quantum circuit to generate the Bell state
    qc = compose_quantum_circuit(num_qubits)

    # Quantum Circuit for Eavesdropping
    qc_eve = compose_quantum_circuit_for_eve(num_qubits)

    for i in range(num_qubits):
        qc.id(i)

    # Eve eavesdrops the qubit Bob will receive 
    qc, eve_basis  = intercept_resend(qc, qc_eve, eve_basis, intercept_prob, num_qubits)

    alice_qubits = [i for i in range(0, num_qubits, 2)]
    bob_qubits = [i for i in range(1, num_qubits, 2)]

    # Apply the quantum error chanel
    noise_model = apply_noise_model_to_qubits(noise_prob, alice_qubits, bob_qubits)

    # 測定（IDゲートにノイズをかけて、Xエラーとして再現）
    qc, alice_bits, bob_bits = alice_bob_measurement(qc, num_qubits, noise_model)


    # eb_basis, eb_matches = check_bases(eve_basis,bob_basis)
    # eb_bits = check_bits(eve_bits,bob_bits,eb_basis)


    ab_basis = check_bases(alice_basis,bob_basis)
    # ab_bits = check_bits(alice_bits, bob_bits, ab_basis)

    # Alice sifted key 
    alice_siftedkey = gen_alicekey(alice_bits, ab_basis)
    # Bob sifted key 
    bob_siftedkey = gen_bobkey(bob_bits, ab_basis)

    return alice_siftedkey, bob_siftedkey


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

# Generate bell state (Need 2 qubits per a state)
def get_bellState(n):
    qc = QuantumCircuit(n,n) 
    for i in range(0, n, 2):
        # i: corresponds to Alice's qubit.
        # i+1: corresponds to Bob's qubit.
        qc.x(i+1) # Pauli-X gate 
        qc.h(i) # Hadamard gate 
        qc.cx(i,i+1) # CNOT gate
        # qc.x(i+1) # Pauli-X gate 
        # qc.h(i) # Hadamard gate
        # qc.z(i) # Pauli-Z gate
        # qc.z(i+1) # Pauli-Z  gate 
        # qc.cx(i,i+1) # CNOT gate
    # print(qc.draw())
    qc.barrier()
    # print(qc)
    return qc

def compose_quantum_circuit(num_qubit) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubit, num_qubit)
    qc.compose(get_bellState(num_qubit), inplace=True)
    return qc


def compose_quantum_circuit_for_eve(num_qubit) -> QuantumCircuit:
    qc2 = QuantumCircuit(num_qubit, num_qubit)
    qc2.compose(get_bellState(num_qubit), inplace=True)
    return qc2


def apply_noise_model(p_meas):
    error_meas = pauli_error([('X', p_meas), ('I', 1 - p_meas)])
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(error_meas, "measure")

    return noise_model


def apply_noise_model_to_qubits(p_meas, alice_qubits, bob_qubits):
    noise_model = NoiseModel()
    
    # 1量子ビット用の測定エラー（例：X誤り）
    error = pauli_error([('X', p_meas), ('I', 1 - p_meas)])
    
    # アリスの各量子ビットに適用
    for qubit in alice_qubits:
        noise_model.add_quantum_error(error, 'measure', [qubit])
        
    # ボブの各量子ビットに適用
    for qubit in bob_qubits:
        noise_model.add_quantum_error(error, 'measure', [qubit])
        
    return noise_model


def alice_bob_measurement(qc, num_qubits, noise_model):
    qc.barrier()
    qc.measure(list(range(num_qubits)), list(range(num_qubits)))

    # IDゲートを挿入することで、特定量子ビットにnoise_modelが作用可能に（Xノイズのトリガー）
    for qubit in range(num_qubits):
        qc.id(qubit)

    result = backend.run(qc, shots=1, noise_model=noise_model).result()
    counts = result.get_counts()
    max_key = max(counts, key=counts.get)
    bits = ''.join(reversed(max_key))
    alice_bits = bits[::2]
    bob_bits = bits[1::2]

    return [qc, alice_bits, bob_bits]


# # Alice and Bob measure own qubits and generate bit
# def alice_bob_measurement(qc, num_qubits, noise_model):

#     qc.measure(list(range(num_qubits)), list(range(num_qubits)))

#     result = backend.run(qc, shots=1, noise_model=noise_model).result()
#     counts = result.get_counts()
#     max_key = max(counts, key=counts.get)
#     bits = ''.join(reversed(max_key))
#     alice_bits = bits[::2]
#     bob_bits = bits[1::2]

#     return [qc, alice_bits, bob_bits]


# check where bases matched
def check_bases(b1,b2):
    check = ''
    # matches = 0
    for i in range(len(b1)):
        if b1[i] == b2[i]: 
            check += "Y" 
            # matches += 1
        else:
            check += "-"
    return check

# check where measurement bits matched
def check_bits(b1,b2,bck):
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

def gen_alicekey(bits, ab_bases):
    alice_sifted_key = ''
    for i in range(len(bits)):
        if ab_bases[i] == 'Y':
            alice_sifted_key += bits[i]
    return alice_sifted_key


def gen_bobkey(bits, ab_bases):
    bob_sifted_key = ''
    for i in range(len(bits)):
        if ab_bases[i] == 'Y':
            # bits[i] を反転
            flipped_bit = '1' if bits[i] == '0' else '0'
            bob_sifted_key += flipped_bit
    return bob_sifted_key


# intercept Alice'squbits to measure and resend to Bob
# 二つの量子回路(qc, qc2)を使う理由
# qc2 is required for Eve to have a sifted key (measurement) based on Alice's Qubit and her own Basis. 
# (Once a qubit is measured, its state is broken, so measuring the same qubit again does not give the original information.)
def intercept_resend(qc, qc2, eve_basis, intercept_prob, num_qubits):
    backend = AerSimulator()

    num_bell = num_qubits // 2
    num_to_intercept = int(num_bell * intercept_prob)
    bell_indices = list(range(num_bell))
    intercepted_bell_indices = random.sample(bell_indices, num_to_intercept)
    eve_basis = list(eve_basis)

    # Eveが盗聴しないペアには '!' をセット
    for i in range(num_bell):
        if i not in intercepted_bell_indices:
            eve_basis[i] = '!'

    # Eveの測定処理（Bob側の量子ビットのみ）
    for bell_index in intercepted_bell_indices:
        bob_qubit_index = bell_index * 2 + 1  # Bobの量子ビットの絶対インデックス
        if eve_basis[bell_index] == '1':
            qc.h(bob_qubit_index)
            qc2.h(bob_qubit_index)

    qc2.measure(list(range(num_qubits)), list(range(num_qubits)))
    result = backend.run(qc2, shots=1).result()
    counts = result.get_counts()
    max_key = max(counts, key=counts.get)
    bits = ''.join(reversed(max_key))

    # 測定結果をもとにBob側を復元
    for bell_index in intercepted_bell_indices:
        bob_qubit_index = bell_index * 2 + 1
        bit = bits[bob_qubit_index]
        if eve_basis[bell_index] == '0':
            if bit == '1':
                qc.x(bob_qubit_index)
        elif eve_basis[bell_index] == '1':
            if bit == '0':
                qc.h(bob_qubit_index)
            else:
                qc.x(bob_qubit_index)
                qc.h(bob_qubit_index)

    qc.barrier()
    return [qc, eve_basis]


import numpy as np
import matplotlib.pyplot as plt
from figure_config import loadMarker, loadColor  # スタイル設定のモジュール名に合わせて変更
import os

def main():
    noise_prob_range = [0.00, 0.02, 0.04, 0.06, 0.08, 0.10]  # 2%ずつ
    intercept_prob_range = [0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
    num_qubits = 250
    count = 100

    qber_results = []  # 2次元リストに保存： [noise][intercept]

    color_cycle = loadColor()
    marker_cycle = loadMarker()

    plt.figure(figsize=(10, 6))

    for noise_prob in noise_prob_range:
        avg_qbers = []  # intercept_probごとの平均QBER
        for intercept_prob in intercept_prob_range:
            total_qber = 0
            for _ in range(count):
                part_ka, part_kb = generate_Siftedkey(user0, user1, num_qubits, noise_prob, intercept_prob)

                if len(part_ka) == 0:
                    continue

                err_num = sum(1 for a, b in zip(part_ka, part_kb) if a != b)
                qber = (err_num / len(part_ka)) * 100
                total_qber += qber

            average_qber = total_qber / count
            avg_qbers.append(average_qber)
            print(f"Noise: {noise_prob:.2f}, Intercept: {intercept_prob:.2f}, QBER: {average_qber:.2f}%")

        qber_results.append(avg_qbers)

        # グラフにプロット
        plt.plot(
            [x * 100 for x in intercept_prob_range],  # 横軸（%）
            avg_qbers,
            label=f'Channel noise = {int(noise_prob*100)}%',
            color=next(color_cycle),
            marker=next(marker_cycle)
        )

    plt.xlabel('Intercept-and-resend ratio (%)')
    plt.ylabel('QBER (%)')
    plt.grid(True)
    plt.legend()
    plt.title('QBER vs Intercept-and-resend ratio')
    plt.tight_layout()

    # 保存
    # os.makedirs('results', exist_ok=True)
    plt.savefig('results/qber_vs_intercept.pdf')
    np.save('results/qber_data.npy', np.array(qber_results))

    print("\n✅ PDF保存: results/qber_vs_intercept.pdf")
    print("✅ データ保存: results/qber_data.npy")

if __name__ == '__main__':
    main()



# print(f'Alice sifted key: {part_ka}')
# print(f'Bob   sifted key: {part_kb}')
# print(f'Number of Error between their sifted key: {err_num}')