"""
Device Architecture for Grid topology
"""

import numpy as np
from environments.environment import Environment


class GridEnvironment(Environment):
    """
    Defines the Device Topology by extending environment
    """

    def __init__(self, rows, columns, circuit, qubit_locations=None):
        """
        Initializes the Grid topology.

        :param circuit: the Circuit object for the environment
        :param qubit_locations: list, initial mapping of logical to physical qubits
        """
        topology = self.generate_grid_topology(rows, columns)
        super().__init__(topology, circuit, qubit_locations)
        self.rows = rows
        self.cols = columns

    @staticmethod
    def generate_grid_topology(rows, columns):
        """
        Add additional links to the grid topology.

        :param rows: number of rows in the grid
        :param columns: number of columns in the grid
        :return: np.array, 2D adjacency matrix
        """
        topology = [[0] * (rows * columns) for _ in range(0, rows * columns)]
        for i in range(0, rows):
            for j in range(0, columns):
                node_index = i*columns + j
                if node_index >= columns:  # up
                    topology[node_index][node_index-columns] = 1
                if node_index < columns*(rows-1):  # down
                    topology[node_index][node_index+columns] = 1
                if node_index % columns > 0:  # left
                    topology[node_index][node_index-1] = 1
                if node_index % columns < columns-1:  # right
                    topology[node_index][node_index+1] = 1
        return np.array(topology)
