# Upgrade qiskit version to 2.0.0
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (NoiseModel, pauli_error)
import time
import random
import socket


count = 1000
sifted_key_length = 1000
num_qubits_linux = 16
backend = AerSimulator()
intercept_prob = 0
noise_prob = 0.1
kr_efficiency = 1.22


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


def generate_Siftedkey(user0, user1, num_qubits):
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
    noise_model = apply_noise_model(noise_prob)

    # Bob measures Alice's qubit
    qc, bob_bits = bob_measurement(qc, bob_basis, noise_model)
    print(qc.draw())

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

    end = time.time()
    runtime = end - start

    sender_classical_channel.close()
    receiver_classical_channel.close()

    return ka, kb, runtime


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


def apply_noise_model(p_meas):
    error_meas = pauli_error([('X', p_meas), ('I', 1 - p_meas)])
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


def main():
    print(f"Channel Noise Ratio:             {noise_prob*100}%")
    print(f"Intercept-and-resend Ratio:      {intercept_prob*100}%")
    part_ka, part_kb, execution_time = generate_Siftedkey(user0, user1, int(1e4))
    print(f'Sifted key: {part_ka} and {part_kb}')
    print(f'Runtime: {execution_time}')


if __name__ == '__main__':
    main()
