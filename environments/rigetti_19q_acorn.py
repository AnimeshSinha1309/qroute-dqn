"""
Device Architecture for Rigetti 19Q Acorn
"""

import numpy as np
from environments.environment import Environment


class Rigetti19QAcorn(Environment):
    """
    Defines the Device Topology by extending environment
    """

    def __init__(self, circuit, qubit_locations=None):
        """
        Initializes the Rigetti 19Q Acorn topology.

        :param circuit: the Circuit object for the environment
        :param qubit_locations: list, initial mapping of logical to physical qubits
        """
        topology = self._generate_acorn_topology()
        super().__init__(topology, circuit, qubit_locations)
        self.rows = 4
        self.cols = 5

    @staticmethod
    def _generate_acorn_topology():
        """
        Add additional links to the grid topology.
        :return: np.array, 2D adjacency matrix
        """
        topology = [[0] * 20 for _ in range(20)]

        links = [(0, 5), (0, 6), (1, 6), (1, 7), (2, 7), (2, 8), (3, 8), (3, 9), (4, 9), (5, 10), (6, 11), (7, 12),
                 (8, 13), (9, 14), (10, 15), (10, 16), (11, 16), (11, 17), (12, 17), (12, 18), (13, 18), (13, 19),
                 (14, 19)]

        for (n1, n2) in links:
            topology[n1][n2] = 1
            topology[n2][n1] = 1

        return np.array(topology)
