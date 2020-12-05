"""
Simulates an actual device while checking for equivalence against the logical circuit
"""


class PhysicalEnvironment:
    """
    Class that takes:
        * Logical Circuit
        * Physical Circuit
        * Device Topology
    And checks for correctness and equivalence
    """

    def __init__(self, original_circuit, scheduled_circuit, environment, initial_qubit_locations, verbose=False):
        """
        Creates the environment in which to run the circuit while checking for correctness

        :param original_circuit: Circuit object, represents the logical circuit
        :param scheduled_circuit: Circuit object, represents the circuit executable in hardware
        :param environment: Environment object, contains the device map
        :param initial_qubit_locations: list, Starting map of the qubits
        :param verbose: bool, whether to log all the outputs
        """
        if verbose:
            print('Original gates:', original_circuit.gates)

        self.circuit = original_circuit.to_dqn_rep()
        self.gates = scheduled_circuit.gates

        self.topology = environment.adjacency_matrix

        self.state = (initial_qubit_locations[:], [0]*original_circuit.n_qubits)  # Qubit locations and circuit progress

        self.verbose = verbose

    def execute_swap(self, n1, n2):
        """
        Checks the execution of the SWAP gate and whether it was expected in the logical circuit

        :param n1: int, index of operand node 1
        :param n2: int, index of operand node 2
        """
        if self.topology[n1][n2] != 1:
            exit('Nodes ' + str(n1) + ' and ' + str(n2) + ' not adjacent while executing SWAP')

        qubit_locations = self.state[0]

        q1 = qubit_locations[n1]
        q2 = qubit_locations[n2]

        qubit_locations[n1] = q2
        qubit_locations[n2] = q1

    def execute_cnot(self, n1, n2):
        """
        Checks the execution of the CNOT gate and whether it was expected in the logical circuit

        :param n1: int, index of operand node 1
        :param n2: int, index of operand node 2
        """
        if self.topology[n1][n2] != 1:
            exit('Nodes ' + str(n1) + ' and ' + str(n2) + ' not adjacent while executing CNOT')

        qubit_locations, circuit_progress = self.state

        q1 = qubit_locations[n1]
        q2 = qubit_locations[n2]

        if not (self.circuit[q1][circuit_progress[q1]] == q2 and self.circuit[q2][circuit_progress[q2]] == q1):
            exit('Qubits ' + str(q1) + ' and ' + str(q2) + ' are not looking to interact')

        circuit_progress[q1] += 1
        circuit_progress[q2] += 1

    def execute_gate(self, n1, n2, gate_type):
        """
        Checks the execution of the arbitrary gate and whether it was expected in the logical circuit
        Uses the check function for the relevant gate, CNOT or SWAP

        :param n1: int, index of operand node 1
        :param n2: int, index of operand node 2
        :param gate_type: Operation of the gate being executed
        :return:
        """
        if self.verbose:
            print('Executing gate:', (n1, n2, gate_type))

        if 'swap' in gate_type.lower():
            self.execute_swap(n1, n2)
        elif 'cx' in gate_type.lower() or 'cnot' in gate_type.lower():
            self.execute_cnot(n1, n2)
        else:
            exit('Unknown gate type "' + str(gate_type) + '" in circuit')

    def execute_circuit(self):
        """
        Looks over and executes the circuit while checking for equivalence and physical feasibility
        """
        if self.verbose:
            print('Executing circuit on physical environment')
            print('Initial qubit locations:', self.state[0])

        for gate_type, n1, n2 in self.gates:
            self.execute_gate(n1, n2, gate_type)

        for q in range(len(self.circuit)):
            if self.state[1][q] != len(self.circuit[q]):
                exit('Circuit not complete')

        if self.verbose:
            print('All gates scheduled')
            print()


def verify_circuit(original_circuit, scheduled_circuit, environment, initial_qubit_locations, verbose=False):
    """
    Verifies the circuit by executing it

    :param original_circuit: Circuit object, represents the logical circuit
    :param scheduled_circuit: Circuit object, represents the circuit executable in hardware
    :param environment: Environment object, contains the device map
    :param initial_qubit_locations: list, Starting map of the qubits
    :param verbose: bool, whether to log all the outputs
    """
    physical_env = PhysicalEnvironment(original_circuit, scheduled_circuit, environment,
                                       initial_qubit_locations, verbose)
    physical_env.execute_circuit()
