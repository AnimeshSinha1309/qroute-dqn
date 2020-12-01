import numpy as np


class State:
    """
    Represents the State of the system when transforming a circuit. This holds the reference
    copy of the environment and the state of the transformation (even within a step)

    :param qubit_locations: The mapping array, tau
    :param qubit_targets: Next qubit location each qubit needs to interact with
    :param circuit_progress: Array keeping track of how many gates are executed by each qubit for updates
    :param protected_nodes: The nodes that are sealed because they are being operation in this step

    :param env: holds all the static information about the state, which does not evolve in time
    """

    def __init__(self, env, qubit_locations=None, qubit_targets=None, circuit_progress=None, protected_nodes=None):
        self.qubit_locations = qubit_locations
        self.qubit_targets = qubit_targets
        self.circuit_progress = circuit_progress
        self.protected_nodes = protected_nodes
        # The state must have access to the overall environment
        self.env = env

    def generate_starting_state(self, circuit=None, qubit_locations=None):
        """

        :param circuit:
        :param qubit_locations:
        :return:
        """
        if circuit is not None:
            self.env.circuit = np.copy(circuit)
        if qubit_locations is None:
            self.qubit_locations = list(np.arange(self.env.number_of_nodes))
            np.random.shuffle(self.qubit_locations)
        else:
            self.qubit_locations = qubit_locations[:]

        self.qubit_targets = [interactions[0] if len(interactions) > 0 else -1 for interactions in self.env.circuit]
        self.circuit_progress = [0] * self.env.number_of_qubits
        gates_to_schedule, self.protected_nodes = self.env.next_gates_to_schedule_between_nodes(
            self.qubit_targets, self.qubit_locations)
        return gates_to_schedule

    def schedule_gates(self):
        """
        Updates the state of the system with whatever interactions can be executed on the hardware.
        This function MUTATES the state.

        :return: int, reward gained by being able to schedule a gate
        """
        reward = 0
        for (q1, q2) in self.next_gates_to_schedule():
            # Increment the progress for both qubits by 1
            self.circuit_progress[q1] += 1
            self.circuit_progress[q2] += 1
            # Updates the qubit targets
            self.qubit_targets[q1] = self.env.circuit[q1][self.circuit_progress[q1]] \
                if self.circuit_progress[q1] < len(self.env.circuit[q1]) else -1
            self.qubit_targets[q2] = self.env.circuit[q2][self.circuit_progress[q2]] \
                if self.circuit_progress[q2] < len(self.env.circuit[q2]) else -1
            # The the reward for this gate which will be executed in next time step for sure, (q1, q2)
            reward += self.env.gate_reward
        return reward

    def next_gates(self):
        """
        For each qubit, it assigns a next interaction if both qubits want to interact with each other.
        Assigns None in the interaction if the qubits do not interact with each other.

        :return gates: list of length n_qubits, (q1, q2) if both want to interact with each other, None otherwise
        """
        gates = [(q, self.qubit_targets[q]) if q == self.qubit_targets[self.qubit_targets[q]] and
                                               q < self.qubit_targets[q]
                 else None for q in range(0, len(self.qubit_targets))]
        return list(filter(lambda gate: gate is not None and gate[0] < gate[1], gates))

    def next_gates_to_schedule(self):
        """
        TODO: Insert a call to hardware check to add directionality together with neighbor as constraint
        Takes the output of next gates, and returns only those which are executable on the hardware

        :return: list, [(q1, q2) where it's the next operation to perform and possible on hardware]
        """
        next_gates = self.next_gates()
        return list(filter(lambda gate: self.env.calculate_gate_distance(gate, self.qubit_locations) == 1, next_gates))

    def next_gates_to_schedule_between_nodes(self):
        """
        Gets the next set of gates we can execute as hardware nodes, and marks them as protected in the state.
        This function MUTATES the state.

        :return: list, [(n1, n2) next gates we can schedule]
        """
        # Converts the qubit-gates to the node gates
        next_gates_to_schedule = self.next_gates_to_schedule()
        next_gates_to_schedule_between_nodes = []
        for (q1, q2) in next_gates_to_schedule:
            (n1, n2) = (np.where(np.array(self.qubit_locations) == q1)[0][0],
                        np.where(np.array(self.qubit_locations) == q2)[0][0])
            gate_between_nodes = (n1, n2) if n1 < n2 else (n2, n1)
            next_gates_to_schedule_between_nodes.append(gate_between_nodes)
        # Makes those nodes as protected which are in the gate arrays
        protected_nodes = set()
        for (n1, n2) in next_gates_to_schedule_between_nodes:
            protected_nodes.add(n1)
            protected_nodes.add(n2)
        # Returns the gate array and updates protected nodes
        self.protected_nodes = protected_nodes
        return next_gates_to_schedule_between_nodes

    def __copy__(self):
        return State(self.env, self.qubit_locations[:], self.qubit_targets[:],
                     self.circuit_progress[:], set(self.protected_nodes))
