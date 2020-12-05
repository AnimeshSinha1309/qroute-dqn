"""
External routing software using Cirq greedy routing
"""

import random

import networkx as nx
import cirq, cirq.contrib.routing.greedy as ccr_greedy

from environments.circuits import NodeCircuit
from utils.circuit_tools import assemble_timesteps_from_gates


def generate_device_graph(environment):
    """
    Generates the edge list for the given environment graph

    :param environment: environment.environment.Environment, our environment object
    :return: list of qubits, network-x graph of device
    """
    device_graph = nx.Graph()
    nodes = [cirq.NamedQubit('node' + str(n)) for n in range(environment.number_of_nodes)]

    a = environment.adjacency_matrix

    for n in nodes:
        device_graph.add_node(n)

    for i in range(len(a)):
        for j in range(len(a[0])):
            if a[i][j] == 1:
                device_graph.add_edge(nodes[i], nodes[j])

    return nodes, device_graph


def convert_circuit_to_cirq_format(circuit, qubits):
    """
    Converts the array of gates to an equivalent cirq circuit

    :param circuit: list of tuple, [(q1, q2)], series of gates as tuples of indices
    :param qubits: array of qubits ordered by index in circuit
    :return: cirq.Circuit with all the CX(q1, q2)
    """
    cirq_circuit = cirq.Circuit()

    for (q1, q2) in circuit.gates:
        cirq_circuit.append([cirq.CX(qubits[q1], qubits[q2])])

    return cirq_circuit


def schedule_swaps(environment, circuit, qubit_locations=None):
    """
    Solves the qubit routing problem using Cirq greedy routing

    :param environment: environment.environment.Environment, our environment object
    :param circuit: environment.circuit.QubitCircuit, our circuit object
    :param qubit_locations: list, mapping array from logical qubits to physical qubits
    :return: layers of the circuit, final depth of the circuit
    """
    unused_qubits = set()

    for q, interactions in enumerate(circuit.to_dqn_rep()):
        if len(interactions) == 0:
            unused_qubits.add(q)

    qubits = [cirq.NamedQubit('qubit' + str(n)) for n in range(circuit.n_qubits)]
    nodes, device_graph = generate_device_graph(environment)

    circuit = convert_circuit_to_cirq_format(circuit, qubits)

    if qubit_locations is None:
        qubit_locations = list(range(environment.number_of_nodes))
        random.shuffle(qubit_locations)

    initial_mapping = {nodes[n]: qubits[q] for n, q in list(filter(
        lambda p: p[1] not in unused_qubits, enumerate(qubit_locations)))}

    swap_network = ccr_greedy.route_circuit_greedily(
        circuit, device_graph, max_search_radius=2, initial_mapping=initial_mapping)
    routed_circuit = swap_network.circuit

    gates = []

    for op in routed_circuit.all_operations():
        op_code = 'SWAP' if 'Swap' in str(op.gate) else 'CNOT'
        n1 = int(op.qubits[0].name.replace('node', ''))
        n2 = int(op.qubits[1].name.replace('node', ''))

        gates.append((op_code, n1, n2))

    cirq_depth = len(routed_circuit.moments)

    node_circuit = NodeCircuit.from_gates(environment.number_of_nodes, gates)

    calculated_depth = node_circuit.depth()

    if cirq_depth != calculated_depth:
        print('Cirq depth:', cirq_depth)
        print('Calculated depth:', calculated_depth)
        exit("Cirq depth disagrees with calculated depth")

    layers = assemble_timesteps_from_gates(node_circuit.n_nodes, node_circuit.gates)

    return layers, cirq_depth
