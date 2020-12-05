"""
External routing software using Qiskit routing
"""

import random

from qiskit.transpiler import CouplingMap
from qiskit.transpiler.passes.routing import StochasticSwap
from qiskit.transpiler.passes.basis.decompose import Decompose
from qiskit.circuit.library import SwapGate
from qiskit.converters import circuit_to_dag, dag_to_circuit

from environments.circuits import NodeCircuit
from environments.physical_environment import verify_circuit
from other_systems.schedulers import assemble_timesteps_from_gates

MethodClass = StochasticSwap


def generate_coupling_map(environment):
    """
    Generates the coupling map from the device adjacency matrix

    :param environment: environment.environment.Environment, our environment object
    :return: list of edges on the device topology (i, j)
    """
    coupling_map = CouplingMap()

    a = environment.adjacency_matrix

    for i in range(len(a)):
        for j in range(len(a[0])):
            if a[i][j] == 1:
                coupling_map.add_edge(i, j)
                coupling_map.add_edge(j, i)

    if not coupling_map.is_symmetric:
        exit("Qiskit coupling map was not symmetric")

    return coupling_map


def schedule_swaps(environment, circuit, qubit_locations=None, safety_checks_on=False, decompose_cnots=False):
    """
    Solves the qubit routing problem using Cirq greedy routing

    :param environment: environment.environment.Environment, our environment object
    :param circuit: environment.circuit.QubitCircuit, our circuit object
    :param qubit_locations: list, mapping array from logical qubits to physical qubits
    :param safety_checks_on: bool
    :param decompose_cnots: bool, whether to decompose SWAP gates to CNOT
    :return: layers of the circuit, final depth of the circuit
    """
    original_circuit = circuit

    if qubit_locations is None:
        qubit_locations = list(range(environment.number_of_nodes))
        random.shuffle(qubit_locations)

    initial_qubit_locations = qubit_locations[:]

    circuit = circuit.to_qiskit_rep(qubit_locations=qubit_locations)
    coupling_map = generate_coupling_map(environment)

    dag_circuit = circuit_to_dag(circuit)

    method_instance = MethodClass(coupling_map, trials=500)
    mapped_dag_circuit = method_instance.run(dag_circuit)
    mapped_circuit = dag_to_circuit(mapped_dag_circuit)

    node_circuit = NodeCircuit.from_qiskit_rep(mapped_circuit, decompose=decompose_cnots)

    if decompose_cnots:
        decomposition_pass = Decompose(type(SwapGate()))
        mapped_dag_circuit = decomposition_pass.run(mapped_dag_circuit)
        mapped_circuit = dag_to_circuit(mapped_dag_circuit)

    qiskit_depth = mapped_circuit.depth()
    calculated_depth = node_circuit.depth()

    if qiskit_depth != calculated_depth:
        print('Data:', mapped_circuit.data)
        print('Qiskit depth:', qiskit_depth)
        print('Calculated depth:', calculated_depth)
        print()

        exit("Qiskit depth disagrees with calculated depth")

    layers = assemble_timesteps_from_gates(node_circuit.n_nodes, node_circuit.gates)

    if safety_checks_on:
        verify_circuit(original_circuit, node_circuit, environment, initial_qubit_locations)

    return layers, qiskit_depth
