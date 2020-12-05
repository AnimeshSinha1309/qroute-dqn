"""
Device Architecture for Google Sycamore
"""

import numpy as np
from environments.environment import Environment


class GoogleSycamore(Environment):
    """
    Defines the Device Topology by extending environment
    """

    def __init__(self, circuit, qubit_locations=None):
        """
        Initializes the Google Sycamore topology.

        :param circuit: the Circuit object for the environment
        :param qubit_locations: list, initial mapping of logical to physical qubits
        """
        self.rows = 9
        self.cols = 6
        topology = self.generate_sycamore_topology(self.rows, self.cols)
        super().__init__(topology, circuit, qubit_locations)

    @staticmethod
    def generate_sycamore_topology(rows, cols):

        topology = [[0] * 54 for _ in range(54)]

        links = []

        for r in range(rows):
            if r % 2 != 0:
                for c in range(cols):
                    n = r * cols + c
                    links += [(n, (r-1)*cols + c), (n, (r+1)*cols + c)]

                    if c < cols-1:
                        links += [(n, (r-1)*cols + c+1), (n, (r+1)*cols + c+1)]

        for (n1, n2) in links:
            topology[n1][n2] = 1
            topology[n2][n1] = 1

        return np.array(topology)
