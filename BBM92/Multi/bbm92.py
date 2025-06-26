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


class Repeater:
   
    def __init__(self, name: str):
        self.name = name
        self.sharekeys = []

    def manage_sharekey(self, new_sharekey: str):
        self.sharekeys.append(new_sharekey)
        print(f"新しい共有鍵が追加されました。現在の鍵リスト: {self.sharekeys}")

class User:
    def __init__(self, name: str):
        self.name = name
        self.sharekeys = ''
    
    def __repr__(self):
        return f"User(name='{self.name}')"


def generate_Siftedkey(num_qubits):
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
    qc, alice_bits, bob_bits = alice_bob_measurement(qc, num_qubits)

    # eb_basis, eb_matches = check_bases(eve_basis,bob_basis)
    # eb_bits = check_bits(eve_bits,bob_bits,eb_basis)

    # user0.create_socket_for_classical()
    # user1.create_socket_for_classical()
    # sender_classical_channel = user0.socket_classical
    # receiver_classical_channel = user1.socket_classical

    # Alice sifted key
    alice_siftedkey=''
    # Bob sifted key
    bob_siftedkey=''
    # Eve sifted key
    # eve_siftedkey=''

    # Announce bob's basis
    # receiver_classical_channel.send(bob_basis.encode('utf-8'))
    # bob_basis = sender_classical_channel.recv(4096).decode('utf-8')
    # Alice's side
    ab_basis = check_bases(alice_basis,bob_basis)
    alice_siftedkey = gen_alicekey(alice_bits, ab_basis)

    # send the result for comparison
    # sender_classical_channel.send(ab_basis.encode('utf-8'))
    # ab_basis = receiver_classical_channel.recv(4096).decode('utf-8')
    bob_siftedkey = gen_bobkey(bob_bits, ab_basis)
    # print(qc.draw())
    end = time.time()
    runtime = end - start

    # sender_classical_channel.close()
    # receiver_classical_channel.close()
    
    return alice_siftedkey, bob_siftedkey, alice_basis, bob_basis


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
def alice_bob_measurement(qc, num_qubits):

    qc.measure(list(range(num_qubits)), list(range(num_qubits)))

    result = backend.run(qc, shots=1).result()
    counts = result.get_counts()
    max_key = max(counts, key=counts.get)
    bits = ''.join(reversed(max_key))
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
def intercept_resend(qc, qc2, eve_basis , intercept_prob, num_qubits):
    backend = AerSimulator()

    l = len(eve_basis)
    num_to_intercept = int(num_qubits * intercept_prob)
    to_intercept = random.sample(range(num_qubits), num_to_intercept)
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
    # result = execute(qc2,backend,shots=1).result() 
    # bits = list(result.get_counts().keys())[0] 
    # bits = ''.join(list(reversed(bits)))

    result = backend.run(qc, shots=1).result()
    counts = result.get_counts()  # `.get_counts(0)` ではなく `.get_counts()` に変更

    # 取得した測定結果の中で最も出現回数が多いものを採用
    max_key = max(counts, key=counts.get)
    bits = ''.join(reversed(max_key))  # ビット列を逆順にして取得
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
    try:
        num_user = int(input('Number of total users (e.g., 3 for Alice + 2 other users): '))
        if num_user < 2:
            print("Please enter a number greater than or equal to 2.")
            return
    except ValueError:
        print("Invalid input. Please enter a number.")
        return

    num_qubits = 100
    repeater = Repeater("repeater")

    # 生成したユーザーオブジェクトを保存するリスト
    users = []

    print("-" * 30)
    # アリス以外のユーザーの数だけループ（例: num_user=3なら2回実行）
    for i in range(num_user - 1):
        # user_1, user_2 ... のように動的にユーザー名を生成
        user_name = f"user_{i + 1}"
        
        # 新しいユーザーオブジェクトを作成
        new_user = User(user_name)
        
        # アリスと新しいユーザーとの間で鍵交換を実行
        alice_siftedkey, user_siftedkey, alice_basis, user_basis = generate_Siftedkey(num_qubits)
        
        # リピーターがアリスの鍵を管理
        repeater.manage_sharekey(alice_siftedkey)
        
        # 新しいユーザーが自身の鍵を保有
        new_user.sharekeys = user_siftedkey
        
        # 作成したユーザーをリストに追加して管理
        users.append(new_user)
        
        # --- 各ユーザーとの鍵交換の結果をその都度表示 ---
        print(f"Key exchange completed with: {new_user.name}")
        print(f"Generated sifted key length for {new_user.name}: {len(user_siftedkey)}")
        print(f"Repeater's (Alice's) Key:      {alice_siftedkey}")
        print(f"{new_user.name}'s Key:         {new_user.sharekeys}")

        print("-" * 30)

    print("\n--- All Processes Finished ---")
    print(f"Total {len(users)} users have been created and exchanged keys with Alice.")
    print("Created users:", users)


if __name__ == '__main__':
    main()