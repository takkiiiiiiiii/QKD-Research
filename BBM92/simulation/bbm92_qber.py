from qiskit import QuantumCircuit, Aer, execute
from qiskit_aer.noise import (NoiseModel, QuantumError, pauli_error, depolarizing_error)
# from kr_Hamming import key_reconciliation_Hamming
from IPython.display import display
from qiskit.tools.visualization import plot_histogram
import numpy as np
import random
import math
import time

# Implement BBM92

count = 1000
sifted_key_length = 1000
num_qubits_linux = 29 # for Linux
num_qubits_mac = 2 # for mac
backend = Aer.get_backend('qasm_simulator')

noise_prob_range = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
intercept_prob_range = [0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1]

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



def generate_Siftedkey(user0, user1, num_qubits):
    start = time.time()
    num_bellState = int(num_qubits/2)
    alice_basis = qrng(num_bellState)
    bob_basis = qrng(num_bellState)
    eve_basis = qrng(num_bellState)

    # Compose the quantum circuit to generate the Bell state
    qc = compose_quantum_circuit(num_qubits)

    # Quantum Circuit for Eve
    # qc_eve = compose_quantum_circuit_for_eve(num_qubits, alice_bits, alice_basis)

    # Eve eavesdrops Alice's qubits
    # qc_alice, eve_basis, eve_bits = intercept_resend(qc_alice, qc_bob, qc_eve, eve_basis, intercept_prob)

    # Apply the quantum error chanel
    # noise_model = apply_noise_model(noise_prob)

    # Alice measure her own qubit
    qc, alice_bits, bob_bits = alice_bob_measurement(qc, alice_basis, bob_basis, num_qubits)

    # eb_basis, eb_matches = check_bases(eve_basis,bob_basis)
    # eb_bits = check_bits(eve_bits,bob_bits,eb_basis)

    user0.create_socket_for_classical()
    user1.create_socket_for_classical()
    sender_classical_channel = user0.socket_classical
    receiver_classical_channel = user1.socket_classical

    # Alice sifted key
    alice_siftedkey=''
    # Bob sifted key
    bob_siftedkey=''
    # Eve sifted key
    # eve_siftedkey=''

    # Announce bob's basis
    receiver_classical_channel.send(bob_basis.encode('utf-8'))
    bob_basis = sender_classical_channel.recv(4096).decode('utf-8')
    # Alice's side
    ab_basis = check_bases(alice_basis,bob_basis)
    ab_bits = check_bits(alice_bits, bob_bits, ab_basis)
    alice_siftedkey = gen_key(alice_bits, ab_basis)

    # send the result for comparison
    sender_classical_channel.send(ab_basis.encode('utf-8'))
    ab_basis = receiver_classical_channel.recv(4096).decode('utf-8')
    bob_siftedkey = gen_key(bob_bits, ab_basis)
    # print(qc.draw())
    end = time.time()
    runtime = end - start

    sender_classical_channel.close()
    receiver_classical_channel.close()
    
    return alice_siftedkey, bob_siftedkey, runtime


def qrng(n):
    # generate n-bit string from measurement on n qubits using QuantumCircuit
    qc = QuantumCircuit(n,n)
    for i in range(n):
        qc.h(i) # The Hadamard gate has the effect of projecting a qubit to a 0 or 1 state with equal probability.
    qc.measure(list(range(n)),list(range(n)))
    # compiled_circuit = transpile(qc, backend)
    # result = backend.run(compiled_circuit, shots=1).result()
    # shot - Number of repetitions of each circuit for sampling
    # Return the results of the job.
    result = execute(qc,backend,shots=1).result() 
    bits = list(result.get_counts().keys())[0]
    bits = ''.join(list(reversed(bits)))
    return bits


# Generate bell state (Need 2 qubits per a state)
def get_bellState(n):
    qc = QuantumCircuit(n,n) 
    for i in range(0, n, 2):
        # i: corresponds to Alice's qubit.
        # i+1: corresponds to Bob's qubit.
        qc.h(i)
        qc.cx(i, i+1)
    # print(qc.draw())
    qc.barrier()
    return qc

# AliceとBobがビット値を生成するための量子回路
def compose_quantum_circuit(num_qubit) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubit, num_qubit)
    qc.compose(get_bellState(num_qubit), inplace=True)
    return qc


# qcと同じ実装だが、イブのビット値を生成するための量子回路
def compose_quantum_circuit_for_eve(num_qubit) -> QuantumCircuit:
    qc2 = QuantumCircuit(num_qubit, num_qubit)
    qc2.compose(get_bellState(num_qubit), inplace=True)
    return qc2


def apply_noise_model(p_meas):
    error_meas = pauli_error([('X', p_meas), ('I', 1 - p_meas)])
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(error_meas, "measure")

    return noise_model

# Alice and Bob measure own qubits and generate bit
def alice_bob_measurement(qc, alice_basis, bob_basis, num_qubits):
    for i in range(num_qubits // 2):
        if alice_basis[i] == '1': 
            qc.h(2 * i)
        if bob_basis[i] == '1': 
            qc.h(2 * i + 1)

    qc.measure(list(range(num_qubits)), list(range(num_qubits)))

    result = execute(qc, backend, shots=1).result()
    counts = result.get_counts()
    max_key = max(counts, key=counts.get)
    bits = max_key[::-1]
    alice_bits = bits[::2]
    bob_bits = bits[1::2]

    return [qc, alice_bits, bob_bits]


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

def gen_key(bits, ab_bases):
    sifted_key = ''  # kaの初期化
    for i in range(len(bits)):
        if ab_bases[i] == 'Y':
            sifted_key += bits[i]
    return sifted_key



# intercept Alice'squbits to measure and resend to Bob
def intercept_resend(qc, qc2, eve_basis , intercept_prob):
    backend = Aer.get_backend('qasm_simulator')

    l = len(eve_basis)
    num_to_intercept = int(num_qubits_linux * intercept_prob)
    to_intercept = random.sample(range(num_qubits_linux), num_to_intercept)
    to_intercept = sorted(to_intercept)
    # print(to_intercept)
    eve_basis = list(eve_basis)

    for i in range(len(eve_basis)):
        if i not in to_intercept:
            eve_basis[i] = '!'

    # print(f"Eve basis: {len(eve_basis)}")

    for i in to_intercept:
        if eve_basis[i] == '1':
            qc.h(i)
            qc2.h(i)

    qc2.measure(list(range(l)),list(range(l))) 
    result = execute(qc2,backend,shots=1).result() 
    bits = list(result.get_counts().keys())[0] 
    bits = ''.join(list(reversed(bits)))

    # qc.reset(list(range(l)))
    
    # イヴの情報を元に、アリスと同じエンコードをして、量子ビットの偏光状態を決める
    for i in range (l):
        if eve_basis[i] == '0':
            if bits[i] == '1':
                qc.x(i)
        elif eve_basis[i] == '1':
            if bits[i] == '0':
                qc.h(i)
            else:
                qc.x(i)
                qc.h(i)

    # display(qc.draw())
    qc.barrier()

    return [qc, eve_basis ,bits]

# execute 1000 times
# Derive the final key rate
# def main():
#     alice_key, bob_key, runtime, alice_basis, bob_basis = generate_Siftedkey(user0, user1, num_qubits_mac)
#     check = check_bases(alice_basis, bob_basis)
#     print(f'Alice key   : {alice_key}')
#     print(f'Alice basis : {alice_basis}')
#     print(f'Alice basis : {check}')
#     print(f'Bob basis   : {bob_basis}')
#     print(f'Bob key     : {bob_key}')

def main():
    noise_prob = 0.07
    intercept_prob = 0.6
    qber = 0
    zero = 0
    print(F"Number of Qubits: {num_qubits_linux}")
    for noise_prob in noise_prob_range: # channel noise
        for intercept_prob in intercept_prob_range:
            print(f"Channel Noise Ratio:             {noise_prob*100}%")
            print(f"Intercept-and-resend Ratio:      {intercept_prob*100}%")
            for i in range(count):
                ka, kb, error_num = generate_Siftedkey(user0, user1, num_qubits_linux, intercept_prob, noise_prob)
                if len(ka) == 0:
                    zero += 1
                    continue
                qber += error_num / len(ka)
            total_qber = qber / (count-zero)
            print(f"Average of QBER({count-zero}times):   {total_qber*100} %")
            qber = 0
            zero = 0

if __name__ == '__main__':
    main()