"""
Environment to provide the logic to step through with swaps and gates.
"""

import numpy as np
import random
import copy

from environments.state import State


class Environment:
    """
    Defined by a Circuit and a Device Topology
    Allows for utility functions and stepping in the environment.
    """

    def __init__(self, topology, circuit, _qubit_locations=None):
        """
        Make the environment, takes a circuit and a device-topology

        :param topology: an adjacency matrix representing the topology of the target system.
        :param circuit: a list of lists representing the circuit to be scheduled.
        :param _qubit_locations: dummy arg received and passed for the inherited classes

        The ith row represents the sequence of interactions that qubit i will undergo during
        the course of the circuit.
        """

        # TODO: check that relevant arguments are indeed NumPy arrays
        # TODO: consider how to deal with circuits that require fewer qubits than available on the target topology

        self.gate_reward = 20
        self.distance_reduction_reward = 2
        self.negative_reward = -10
        self.circuit_completion_reward = 100

        self.alternative_reward_delivery = False

        self.number_of_nodes = len(topology)
        self.number_of_qubits = len(circuit)
        self.adjacency_matrix = np.copy(topology)
        self.circuit = np.copy(circuit) if circuit is not None else None

        self.edge_list = self.generate_edge_list()
        self.distance_matrix = self.generate_distance_matrix()

    @staticmethod
    def generate_random_circuit(number_of_qubits, number_of_gates):
        """
        Makes a random circuit starting

        :param number_of_qubits: count of qubits to build a circuit on
        :param number_of_gates: count of gates in the circuit
        :return: circuit in the DQN representation, list of lists
        """
        circuit = []
        for _ in range(number_of_qubits):
            circuit.append([])
        for _ in range(number_of_gates):
            q1 = random.randint(0, number_of_qubits-1)
            q2 = random.randint(0, number_of_qubits-1)
            while q1 == q2:
                q1 = random.randint(0, number_of_qubits-1)
                q2 = random.randint(0, number_of_qubits-1)
            circuit[q1].append(q2)
            circuit[q2].append(q1)
        return circuit

    def generate_starting_state(self, circuit=None, qubit_locations=None):
        """
        Get's the starting state for the environment.

        :param circuit: list of lists, DQN representation of the circuit, None if we want to retain the same
        :param qubit_locations: list, the mapping of qubit locations
        :return: (State, list), (initial_state, [(n1, n2) next gates we can schedule])
        """
        state = State(env=self)
        gates_scheduled = state.generate_starting_state(circuit, qubit_locations)
        return state, gates_scheduled

    def generate_edge_list(self):
        """
        Gets the list of edges on the hardware

        :return: list of edges
        """
        temp = np.where(self.adjacency_matrix == 1)
        return sorted(list(filter(lambda edge: edge[0] < edge[1], zip(temp[0], temp[1]))))

    def generate_distance_matrix(self):
        """
        TODO: Move this method to the Device modules
        Uses the Floyd-Warshall algorithm to generate a matrix of distances
        between physical nodes in the target topology.

        :return: np.array, 2D, all pairs distances
        """

        dist = np.full((self.number_of_nodes, self.number_of_nodes), np.inf)

        for (u, v) in self.edge_list:
            dist[u][v] = 1
            dist[v][u] = 1

        for v in range(0, self.number_of_nodes):
            dist[v][v] = 0

        for k in range(0, self.number_of_nodes):
            for i in range(0, self.number_of_nodes):
                for j in range(0, self.number_of_nodes):
                    if dist[i][j] > dist[i][k] + dist[k][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
        return dist

    def calculate_gate_distance(self, gate, qubit_locations):
        """
        Gives the physical distance between two qubits given the mapping.

        :param gate: tuple (q1, q2), representing the gate we are measuring the distance of
        :param qubit_locations: list, the mapping
        :return: int
        """
        (q1, q2) = gate
        node1 = np.where(np.array(qubit_locations) == q1)[0][0]
        node2 = np.where(np.array(qubit_locations) == q2)[0][0]
        return self.distance_matrix[node1][node2]

    def calculate_distances(self, qubit_locations, qubit_targets):
        """
        Get's all the distances for each qubits with next operation qubit

        :param qubit_locations: list/array, current mapping of logical to physical qubits
        :param qubit_targets: list/array, the next elements to match against
        :return: list, distances for each qubit on the next operation
        """
        distances = [0] * self.number_of_qubits
        for q in range(self.number_of_qubits):
            target_qubit = qubit_targets[q]
            if target_qubit == -1:
                distances[q] = np.inf
                continue
            node = np.where(np.array(qubit_locations) == q)[0][0]
            target_node = np.where(np.array(qubit_locations) == qubit_targets[q])[0][0]
            distances[q] = self.distance_matrix[node][target_node]
        return distances

    def step(self, action, input_state: State):
        """
        Takes one step in the environment

        :param action: list of bool, whether we want to swap on each of the hardware connected nodes
        :param input_state: State, the state in the previous step
        :return: State, the state in the upcoming step
        """
        state: State = copy.copy(input_state)
        pre_swap_reward = state.schedule_gates()  # can serve reward here
        pre_swap_distances = self.calculate_distances(state.qubit_locations, state.qubit_targets)
        swap_edge_indices = np.where(np.array(action) == 1)[0]
        swap_edges = [self.edge_list[i] for i in swap_edge_indices]

        for (node1, node2) in swap_edges:
            state.qubit_locations[node1], state.qubit_locations[node2] = \
                state.qubit_locations[node2], state.qubit_locations[node1]
        post_swap_distances = self.calculate_distances(state.qubit_locations, state.qubit_targets)
        distance_reduction_reward = 0

        for q in range(self.number_of_qubits):
            if post_swap_distances[q] < pre_swap_distances[q]:
                distance_reduction_reward += self.distance_reduction_reward
        gates_scheduled = state.next_gates_to_schedule_between_nodes()
        post_swap_reward = len(gates_scheduled) * self.gate_reward

        reward = pre_swap_reward if self.alternative_reward_delivery else post_swap_reward + distance_reduction_reward
        next_state = copy.copy(state)
        return next_state, reward, next_state.is_done(), gates_scheduled

    def get_neighbour_edge_nums(self, edge_num):
        """
        Finds edges that share a node with input edge.

        :param edge_num: index of input edge (used to get input edge from self.edge_list)
        :return: neighbour_edge_nums: indices of neighbouring edges.
        """
        node1, node2 = self.edge_list[edge_num]
        neighbour_edge_nums = []
        for edge in self.edge_list:
            if node1 in edge or node2 in edge:
                neighbour_edge_nums.append(self.edge_list.index(edge))

        neighbour_edge_nums.remove(edge_num)
        return neighbour_edge_nums
