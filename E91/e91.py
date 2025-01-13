from qiskit import QuantumCircuit, Aer, QuantumRegister, ClassicalRegister, execute
from qiskit_aer.noise import (NoiseModel, QuantumError, pauli_error, depolarizing_error)
# from kr_Hamming import key_reconciliation_Hamming
from IPython.display import display
from qiskit.tools.visualization import plot_histogram
import numpy as np
import random
import math
import time
import re
from IPython.display import display


# Implement BBM92

count = 1000
sifted_key_length = 1000
num_qubits_linux = 29 # for Linux
num_qubits_mac = 2 # for mac
backend = Aer.get_backend('qasm_simulator')


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



def generate_Siftedkey(user0, user1, numberOfSinglets):

    qr = QuantumRegister(2, name="qr")
    cr = ClassicalRegister(4, name="cr")
    # num_bellState = int(numberOfSinglets)
    # alice_basis = qrng(numberOfSinglets)
    # bob_basis = qrng(numberOfSinglets)
    # eve_basis = qrng(numberOfSinglets)

    # Compose the quantum circuit to generate the singlet state
    singlet = compose_quantum_circuit(qr, cr)

    # Alice measure her own qubit
    # qc, alice_bits, bob_bits = alice_bob_measurement(qc, alice_basis, bob_basis, num_qubits)
    
    # Alice and Bob create circuit for measurement
    aliceMeasurements, bobMeasurements = create_measurement_circuits(qr, cr)

    # Alice chooses as many bases as there are Singlet states.
    aliceMeasurementChoices = [random.randint(1, 3) for i in range(numberOfSinglets)]

    # Bob chooses as many bases as there are Singlet states.
    bobMeasurementChoices = [random.randint(1, 3) for i in range(numberOfSinglets)] 

    circuits = [] # the list in which the created circuits will be stored
    
    for i in range(numberOfSinglets):
        # create the joint measurement circuit
        # start with the singlet state circuit
        jointCircuit = singlet.copy()
        
        # add Alice's measurement circuit
        jointCircuit.compose(aliceMeasurements[aliceMeasurementChoices[i]-1], inplace=True)
        
        # add Bob's measurement circuit
        jointCircuit.compose(bobMeasurements[bobMeasurementChoices[i]-1], inplace=True)
        
        # add the created circuit to the circuits list
        circuits.append(jointCircuit)


    for i in range(numberOfSinglets):
        display(circuits[i])

    # print(circuits[0].name)
    result = execute(circuits, backend=backend, shots=1).result()


    abPatterns = [
        re.compile('..00$'), # search for the '..00' output (Alice obtained 0 and Bob obtained 0)
        re.compile('..01$'), # search for the '..01' output
        re.compile('..10$'), # search for the '..10' output (Alice obtained -1 and Bob obtained 1)
        re.compile('..11$')  # search for the '..11' output
    ]

    aliceResults = [] # Alice's results (string a)
    bobResults = [] # Bob's results (string a')

    user0.create_socket_for_classical()
    user1.create_socket_for_classical()
    sender_classical_channel = user0.socket_classical
    receiver_classical_channel = user1.socket_classical

    for i in range(numberOfSinglets):

        res = list(result.get_counts(circuits[i]).keys())[0] # extract the key from the dict and transform it to str; execution result of the i-th circuit
        print(f'{i}: {res}')
        
        if abPatterns[0].search(res): # check if the key is '..00' (if the measurement results are 0,0)
            aliceResults.append(0) # Alice got the result 0
            bobResults.append(0) # Bob got the result 0
        if abPatterns[1].search(res): # check if the key is '..01'
            aliceResults.append(1)
            bobResults.append(0)
        if abPatterns[2].search(res): # check if the key is '..10' (if the measurement results are 0,0)
            aliceResults.append(0) # Alice got the result 0
            bobResults.append(1) # Bob got the result 1
        if abPatterns[3].search(res): # check if the key is '..11'
            aliceResults.append(1)
            bobResults.append(1)



    aliceKey = [] # Alice's key string k
    bobKey = [] # Bob's key string k'

    for i in range(numberOfSinglets):
    # if Alice and Bob have measured the spin projections onto the a_2/b_1 or a_3/b_2 directions
        if (aliceMeasurementChoices[i] == 2 and bobMeasurementChoices[i] == 1) or (aliceMeasurementChoices[i] == 3 and bobMeasurementChoices[i] == 2):
            aliceKey.append(aliceResults[i]) # record the i-th result obtained by Alice as the bit of the secret key k
            bobKey.append(bobResults[i]) # record the multiplied by -1 i-th result obtained Bob as the bit of the secret key k'

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
    # ab_basis = check_bases(alice_basis,bob_basis)
    # ab_bits = check_bits(alice_bits, bob_bits, ab_basis)
    # alice_siftedkey = gen_key(alice_bits, ab_basis)

    # send the result for comparison
    # sender_classical_channel.send(ab_basis.encode('utf-8'))
    # ab_basis = receiver_classical_channel.recv(4096).decode('utf-8')
    # bob_siftedkey = gen_key(bob_bits, ab_basis)
    # print(qc.draw())
    end = time.time()

    # sender_classical_channel.close()
    # receiver_classical_channel.close()
    
    corr = chsh_corr(result, numberOfSinglets, circuits, aliceMeasurementChoices, bobMeasurementChoices, abPatterns) 

    keyLength = len(aliceKey) # length of the sifted key
    abKeyMismatches = 0 # number of mismatching bits in Alice's and Bob's keys
    for j in range(keyLength):
        if aliceKey[j] != bobKey[j]:
            abKeyMismatches += 1

    return aliceKey, bobKey, abKeyMismatches, corr


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


# Generate bell state (singlet state) 
def get_bellState(n):
    qc = QuantumCircuit(n,n) 
    for i in range(0, n, 2):
        # i: corresponds to Alice's qubit.
        # i+1: corresponds to Bob's qubit.
        qc.x(i)
        qc.x(i+1)
        qc.h(i)
        qc.cx(i, i+1)
    # print(qc.draw())
    qc.barrier()
    return qc

# AliceとBobがビット値を生成するための量子回路
def compose_quantum_circuit(qr, cr) -> QuantumCircuit:
    singlet = QuantumCircuit(qr, cr, name='singlet')
    singlet.x(qr[0])
    singlet.x(qr[1])
    singlet.h(qr[0])
    singlet.cx(qr[0],qr[1])
    

    return singlet


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


def create_measurement_circuits(qr, cr): 
    
    ## Alice's measurement circuits

    # measure the spin projection of Alice's qubit onto the a_1 direction (X basis)
    measureA1 = QuantumCircuit(qr, cr, name='measureA1')
    measureA1.h(qr[0])
    measureA1.measure(qr[0],cr[0])

    # measure the spin projection of Alice's qubit onto the a_2 direction (W basis)
    measureA2 = QuantumCircuit(qr, cr, name='measureA2')
    measureA2.s(qr[0])
    measureA2.h(qr[0])
    measureA2.t(qr[0])
    measureA2.h(qr[0])
    measureA2.measure(qr[0],cr[0])

    # measure the spin projection of Alice's qubit onto the a_3 direction (standard Z basis)
    measureA3 = QuantumCircuit(qr, cr, name='measureA3')
    measureA3.measure(qr[0],cr[0])

    ## Bob's measurement circuits

    # measure the spin projection of Bob's qubit onto the b_1 direction (W basis)
    measureB1 = QuantumCircuit(qr, cr, name='measureB1')
    measureB1.s(qr[1])
    measureB1.h(qr[1])
    measureB1.t(qr[1])
    measureB1.h(qr[1])
    measureB1.measure(qr[1],cr[1])

    # measure the spin projection of Bob's qubit onto the b_2 direction (standard Z basis)
    measureB2 = QuantumCircuit(qr, cr, name='measureB2')
    measureB2.measure(qr[1],cr[1])

    # measure the spin projection of Bob's qubit onto the b_3 direction (V basis)
    measureB3 = QuantumCircuit(qr, cr, name='measureB3')
    measureB3.s(qr[1])
    measureB3.h(qr[1])
    measureB3.tdg(qr[1])
    measureB3.h(qr[1])
    measureB3.measure(qr[1],cr[1])

    aliceMeasurements = [measureA1, measureA2, measureA3]
    bobMeasurements = [measureB1, measureB2, measureB3]

    

    return aliceMeasurements, bobMeasurements



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


# function that calculates CHSH correlation value
def chsh_corr(result, numberOfSinglets, circuits, aliceMeasurementChoices, bobMeasurementChoices, abPatterns):
    
    # lists with the counts of measurement results
    # each element represents the number of (-1,-1), (-1,1), (1,-1) and (1,1) results respectively
    countA1B1 = [0, 0, 0, 0] # XW observable
    countA1B3 = [0, 0, 0, 0] # XV observable
    countA3B1 = [0, 0, 0, 0] # ZW observable
    countA3B3 = [0, 0, 0, 0] # ZV observable

    for i in range(numberOfSinglets):

        res = list(result.get_counts(circuits[i]).keys())[0]

        # if the spins of the qubits of the i-th singlet were projected onto the a_1/b_1 directions
        if (aliceMeasurementChoices[i] == 1 and bobMeasurementChoices[i] == 1):
            for j in range(4):
                if abPatterns[j].search(res):
                    countA1B1[j] += 1

        if (aliceMeasurementChoices[i] == 1 and bobMeasurementChoices[i] == 3):
            for j in range(4):
                if abPatterns[j].search(res):
                    countA1B3[j] += 1

        if (aliceMeasurementChoices[i] == 3 and bobMeasurementChoices[i] == 1):
            for j in range(4):
                if abPatterns[j].search(res):
                    countA3B1[j] += 1
                    
        # if the spins of the qubits of the i-th singlet were projected onto the a_3/b_3 directions
        if (aliceMeasurementChoices[i] == 3 and bobMeasurementChoices[i] == 3):
            for j in range(4):
                if abPatterns[j].search(res):
                    countA3B3[j] += 1
                    
    # number of the results obtained from the measurements in a particular basis
    total11 = sum(countA1B1)
    total13 = sum(countA1B3)
    total31 = sum(countA3B1)
    total33 = sum(countA3B3)    

    print(f'total11: {total11}')
    print(f'total13: {total13}')
    print(f'total31: {total31}')
    print(f'total33: {total33}')
                    
    # expectation values of XW, XV, ZW and ZV observables (2)
    if total11 == 0:
        expect11 = 0                 
    else:
        expect11 = (countA1B1[0] - countA1B1[1] - countA1B1[2] + countA1B1[3])/total11 # -1/sqrt(2)

    if total13 == 0:
        expect13 = 0
    else:
        expect13 = (countA1B3[0] - countA1B3[1] - countA1B3[2] + countA1B3[3])/total13 # 1/sqrt(2)

    if total31 == 0:
        expect31 = 0
    else:
        expect31 = (countA3B1[0] - countA3B1[1] - countA3B1[2] + countA3B1[3])/total31 # -1/sqrt(2)
    
    if total33 == 0:
        expect33 = 0
    else:
        expect33 = (countA3B3[0] - countA3B3[1] - countA3B3[2] + countA3B3[3])/total33 # -1/sqrt(2) 
        
    corr = expect11 - expect13 + expect31 + expect33 # calculate the CHSC correlation value (3)
    
    return corr









# execute 1000 times
# Derive the final key rate
def main():
    # alice_key, bob_key, runtime, alice_basis, bob_basis = generate_Siftedkey(user0, user1, num_qubits_mac)
    # check = check_bases(alice_basis, bob_basis)
    # print(f'Alice key   : {alice_key}')
    # print(f'Alice basis : {alice_basis}')
    # print(f'Alice basis : {check}')
    # print(f'Bob basis   : {bob_basis}')
    # print(f'Bob key     : {bob_key}')
    numberOfSinglets = 2
    # qc = get_bellState(2)
    # print(qc.draw())
    aliceKey, bobKey, abKeyMismatches, corr= generate_Siftedkey(user0, user1, numberOfSinglets)
    print('CHSH correlation value: ' + str(round(corr, 3)))
    # Keys
    # print('Length of the key: ' + int(len(aliceKey)))
    print('Number of mismatching bits: ' + str(abKeyMismatches) + '\n')
    print('Alice key: ' + ''.join(str(aliceKey)))
    print('Bob key: '   + ''.join(str(bobKey)))

    




if __name__ == '__main__':
    main()