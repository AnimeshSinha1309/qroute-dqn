"""
Device Architecture for IBM-Q20 Tokyo
"""

from environments.grid_environment import GridEnvironment


class IBMQ20Tokyo(GridEnvironment):
    """
    Defines the Device Topology by extending environment
    """

    def __init__(self, circuit, qubit_locations=None):
        """
        Initializes the IBM-Q20 Tokyo.

        :param circuit: the Circuit object for the environment
        :param qubit_locations: list, initial mapping of logical to physical qubits
        """
        self.rows, self.cols = 4, 5
        topology = self.generate_grid_topology(self.rows, self.cols)
        self._adjust_topology(topology)
        super().__init__(topology, circuit, qubit_locations)

    @staticmethod
    def _adjust_topology(topology):
        """
        Add additional links to the grid topology.

        :param topology: matrix of distances, sets them to 1 if direct link exists
        :return: np.array, 2D adjacency matrix
        """
        bonus_links = [(1, 7), (2, 6), (3, 9), (4, 8), (5, 11), (6, 10), (7, 13),
                       (8, 12), (11, 17), (12, 16), (13, 19), (14, 18)]

        for (n1, n2) in bonus_links:
            topology[n1][n2] = 1
            topology[n2][n1] = 1
