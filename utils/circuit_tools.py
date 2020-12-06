"""
Utility modules to manage random circuit initializations and getting circuit properties
like depth, etc.
"""

import numpy as np
import random

from environments.circuits import QubitCircuit


# GENERIC TOOLS


def calculate_circuit_depth(number_of_nodes, gates):
    """
    Gets the depth of the given circuit

    :param number_of_nodes: number of qubits on device
    :param gates: [(gate_type, n1, n2)], list of operations
    :return: depth of the circuit
    """
    d = [0] * number_of_nodes

    for (_, n1, n2) in gates:
        d_max = max(d[n1], d[n2])

        d[n1] = d_max + 1
        d[n2] = d_max + 1

    return max(d)


def assemble_timesteps_from_gates(number_of_nodes, gates):
    """
    Compute the depth given a circuit

    :param number_of_nodes: int, number of qubits
    :param gates: [(q1, q2)], the list of gates
    :return: int, number of timesteps, i.e. circuit depth
    """
    d = [0] * number_of_nodes
    timesteps = []

    for (gate_type, n1, n2) in gates:
        d_max = max(d[n1], d[n2])

        new_depth = d_max + 1

        d[n1] = new_depth
        d[n2] = new_depth

        if new_depth > len(timesteps):
            timesteps.append([(gate_type, n1, n2)])
        else:
            timesteps[new_depth-1].append((gate_type, n1, n2))

    return timesteps


def print_qiskit_circuit(n_rows, n_cols, gates, qubit_to_node_map):
    """
    Prints the mapping on the grid of device for each time-step

    :param n_rows: int, number of rows on device's grid topology
    :param n_cols: int, number of columns on device's grid topology
    :param gates: [(gate_type, n1, n2)], list of operations
    :param qubit_to_node_map: map, mapping qubits to nodes, initially
    """
    timesteps = assemble_timesteps_from_gates(n_rows * n_cols, gates)

    qubit_locations = np.array([-1] * (n_rows * n_cols))

    for q, n in enumerate(qubit_to_node_map):
        qubit_locations[n] = q

    print(np.reshape(qubit_locations, (n_rows, n_cols)))
    print()

    for t in timesteps:
        for (gate_type, n1, n2) in t:
            if gate_type == 'SwapGate':
                temp = qubit_locations[n1]
                qubit_locations[n1] = qubit_locations[n2]
                qubit_locations[n2] = temp
            else:
                qubit_locations[n1] = -1
                qubit_locations[n2] = -1

        print(np.reshape(qubit_locations, (n_rows, n_cols)))
        print()


# CIRCUIT GENERATION TOOLS


def generate_full_layer_circuit(n_qubits):
    """
    Generates a Circuit object (in our framework) with all pairs interactions

    :param n_qubits: number of qubits
    :return: QubitCircuit, a circuit with all pairs interactions
    """
    circuit = QubitCircuit(n_qubits)

    for i in range(int(n_qubits/2)):
        circuit.cnot(i*2, (i*2)+1)

    return circuit


def generate_completely_random_circuit(n_qubits, n_gates):
    """
    Generates a Circuit with random gates

    :param n_qubits: int, number of qubits
    :param n_gates: int, number of random gates to add
    :return: QubitCircuit, randomly connected
    """
    circuit = QubitCircuit(n_qubits)

    for _ in range(n_gates):
        q1 = random.randint(0, n_qubits-1)
        q2 = random.randint(0, n_qubits-1)

        while q1 == q2:
            q1 = random.randint(0, n_qubits-1)
            q2 = random.randint(0, n_qubits-1)

        circuit.cnot(q1, q2)

    return circuit


def add_layer(circuit, layer_density=1.0):
    """
    Adds a random layer to the circuit, parametrized by density.

    :param circuit: QubitCircuit, the circuit to add the layer to
    :param layer_density: float, fraction of interaction pairs which will have gates on it
    :return: QubitCircuit with the random layer added
    """
    n_qubits = circuit.n_qubits
    n_gates = int(n_qubits/2)

    qubits = list(range(n_qubits))
    random.shuffle(qubits)

    gates = [(qubits[i*2], qubits[(i*2)+1]) for i in range(n_gates)]

    n_gates_to_add = int(n_gates*layer_density)

    for (q1, q2) in gates[:n_gates_to_add]:
        circuit.cnot(q1, q2)


def generate_multi_layer_circuit(n_qubits, n_layers, layer_density=1.0):
    """
    Generates a random circuit, parametrized by density.

    :param n_qubits: int, number of qubits in the circuit
    :param n_layers: int, number of layers in the circuit
    :param layer_density: float, fraction of interaction pairs which will have gates on it
    :return: QubitCircuit, randomly initialized with n_layers and density gates on
    """
    circuit = QubitCircuit(n_qubits)

    for _ in range(n_layers):
        add_layer(circuit, layer_density)

    return circuit
