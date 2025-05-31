# Ref https://quantumcomputinguk.org/tutorials/introduction-to-bell-states

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
def firstBellState(num_qubits):
    circuit = QuantumCircuit(num_qubits,num_qubits)

    circuit.h(0) # Hadamard gate 
    circuit.cx(0,1) # CNOT gate
    circuit.measure(list(range(num_qubits)),list(range(num_qubits))) # Qubit Measurment
    print(circuit)

    job_result = backend.run(circuit, shots=1).result()
    counts = job_result.get_counts()

    print(counts)


def secondBellState(num_qubits):
    circuit = QuantumCircuit(num_qubits,num_qubits)

    circuit.x(0) # Pauli-X gate 
    circuit.h(0) # Hadamard gate 
    circuit.cx(0,1) # CNOT gate
    circuit.measure(list(range(num_qubits)),list(range(num_qubits))) # Qubit Measurment

    print(circuit)

    job_result = backend.run(circuit, shots=1).result()
    counts = job_result.get_counts()

    print(counts)

def thirdBellState(num_qubits):
    circuit = QuantumCircuit(num_qubits,num_qubits)

    circuit.x(1) # Pauli-X gate 
    circuit.h(0) # Hadamard gate 
    circuit.cx(0,1) # CNOT gate
    circuit.measure(list(range(num_qubits)),list(range(num_qubits))) # Qubit Measurment

    print(circuit)

    job_result = backend.run(circuit, shots=1).result()
    counts = job_result.get_counts()

    print(counts)

def fourthBellState(num_qubits):
    circuit = QuantumCircuit(num_qubits,num_qubits)

    circuit.x(1) # Pauli-X gate 
    circuit.h(0) # Hadamard gate
    circuit.z(0) # Pauli-Z gate
    circuit.z(1) # Pauli-Z  gate 
    circuit.cx(0,1) # CNOT gate
    circuit.measure(list(range(num_qubits)),list(range(num_qubits))) # Qubit Measurment

    print(circuit)
    
    job_result = backend.run(circuit, shots=1).result()
    counts = job_result.get_counts()

    print(counts)

def main():
    num_qubits = 2
    print("Creating first Bell State:\n")
    firstBellState(num_qubits)
    print("\nCreating second Bell State:\n")
    secondBellState(num_qubits)
    print("\nCreating third Bell State:\n")
    thirdBellState(num_qubits)
    print("\nCreating fourth Bell State:\n")
    fourthBellState(num_qubits)


if __name__ == '__main__':
    main()