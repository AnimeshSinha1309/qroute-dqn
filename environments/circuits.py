"""
Implements 2 types of circuit classes:
    * QubitCircuit: simple circuit with qubits and ops which are 2 qubit gates
    * NodeCircuit: circuit which also maintains type of op and can decompose swap to 3 CX/CNOT
These classes should implement easy interoperability across Cirq, Qiskit, Circuit-List and DQN representation
(DQN representation is a list of n_qubit lists, each sublist is the ordered set of q_j q_i is operated with)
"""

from qiskit import QuantumCircuit as QiskitCircuit


class QubitCircuit:
    """
    Represents a quantum circuit
    """

    # BUILDERS

    def __init__(self, n_qubits):
        """
        Initialize a circuit with qubits and no operations (as of now)
        :param n_qubits: number of qubits
        """
        self.n_qubits = n_qubits
        self.gates = []

    @staticmethod
    def from_gates(n_qubits, gates):
        """
        Build a Quantum Circuit from a list of gates
        :param n_qubits: number of qubits
        :param gates: list of operations as tuples (qubits...)
        :return: QubitCircuit, the circuit that was built
        """
        circuit = QubitCircuit(n_qubits)
        circuit.gates.extend(gates)
        return circuit

    # GATES

    def cnot(self, q1, q2):
        """
        Appends a CNOT gate to the circuit
        :param q1: operand 1, the control
        :param q2: operand 2, the operated
        :return:
        """
        if q1 >= self.n_qubits or q2 >= self.n_qubits:
            raise Exception('Tried to add a gate ' + str((q1, q2)) +
                            ' but circuit only has ' + str(self.n_qubits) + ' qubits')

        self.gates.append((q1, q2))

    # OTHER METHODS

    def depth(self):
        """
        Computes the depth of the circuit
        :return: int, the depth of the circuit
        """
        d = [0] * self.n_qubits
        for (q1, q2) in self.gates:
            d_max = max(d[q1], d[q2])
            d[q1] = d_max + 1
            d[q2] = d_max + 1
        return max(d)

    # REP GENERATION

    def to_dqn_rep(self):
        """
        Converts the circuit to it's DQN form, which is n_qubits lists,
        each list is the ordered set of operations which each qubit operates with
        :return: list of lists, the DQN representation of the circuit
        """
        dqn_rep = []

        for _ in range(self.n_qubits):
            dqn_rep.append([])

        for (q1, q2) in self.gates:
            dqn_rep[q1].append(q2)
            dqn_rep[q2].append(q1)

        return dqn_rep

    def to_qiskit_rep(self, qubit_locations=None):
        """
        Makes a qiskit circuit out of the input circuit
        :param qubit_locations: mapping (permutation) of qubits, None if identity
        :return: QiskitCircuit object
        """
        if qubit_locations is None:
            gates = self.gates
        else:
            qubit_to_node_map = [-1]*self.n_qubits

            for n, q in enumerate(qubit_locations):
                qubit_to_node_map[q] = n

            gates = list(map(lambda g: (qubit_to_node_map[g[0]], qubit_to_node_map[g[1]]), self.gates))

        qiskit_rep = QiskitCircuit(self.n_qubits)

        for (n1, n2) in gates:
            qiskit_rep.cnot(n1, n2)

        return qiskit_rep


class NodeCircuit:
    """
    Essentially the same as QubitCircuit, but can maintain the type of gate used,
    so models breaking down triplet swap gates to 3 CNOT/CX primitives
    """

    def __init__(self, n_nodes):
        """
        Initializes the circuit
        :param n_nodes: number of nodes
        """
        self.n_nodes = n_nodes
        self.gates = []

    # BUILDERS

    @staticmethod
    def from_gates(n_nodes, gates, decompose=False):
        """
        Generates a circuit from a list of gates
        :param n_nodes: number of qubits
        :param gates: list of 3-tuples, (op, qubit_1, qubit_2) representing operations
        :param decompose: whether to decompose SWAP into 3 CNOT/CX
        :return: NodeCircuit object, which is the resulting circuit
        """
        circuit = NodeCircuit(n_nodes)

        if decompose:
            gates = circuit.decompose_gates(gates)

        circuit.gates.extend(gates)

        return circuit

    @staticmethod
    def from_qiskit_rep(qiskit_rep, decompose=False):
        """
        Generates a circuit from qiskit object
        :param qiskit_rep: input qiskit circuit
        :param decompose: whether to decompose SWAP into 3 CNOT/CX
        :return: NodeCircuit object, which is the resulting circuit
        """
        circuit = NodeCircuit(len(qiskit_rep.qubits))

        gates = []

        for gate_obj, qubits, _ in qiskit_rep.data:
            gate = (gate_obj.__class__.__name__, qubits[0].index, qubits[1].index)
            gates.append(gate)

        if decompose:
            gates = circuit.decompose_gates(gates)

        circuit.gates.extend(gates)

        return circuit

    # OTHER METHODS

    @staticmethod
    def decompose_gates(gates):
        """
        Decomposes Swaps to 3 CNOT, and leaves other supported gates as is (CX and CNOT)
        :param gates: a list of gates to search and decompose swaps in
        :return: the list of decomposed gates
        """
        decomposed_gates = []
        for (op, n1, n2) in gates:
            if 'swap' in op.lower():
                decomposition = [('CnotGate', n1, n2), ('CnotGate', n2, n1), ('CnotGate', n1, n2)]
                decomposed_gates.extend(decomposition)
            elif 'cx' in op.lower() or 'cnot' in op.lower():
                decomposed_gates.append((op, n1, n2))
            else:
                exit('Unknown gate type "' + str(op) + '" in circuit when decomposing')

        return decomposed_gates

    def depth(self):
        """
        Computes the depth of the circuit
        :return: int, the depth of the circuit
        """
        d = [0] * self.n_nodes
        for (_, n1, n2) in self.gates:
            d_max = max(d[n1], d[n2])
            d[n1] = d_max + 1
            d[n2] = d_max + 1
        return max(d)
