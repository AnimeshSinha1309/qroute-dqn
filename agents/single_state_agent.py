import numpy as np
import random

from agents.meta_agent import MetaDQNAgent
from annealers.single_state_annealer import SimpleDQNAnnealer


class SimpleDQNAgent(MetaDQNAgent):

    def __init__(self, environment, memory_size=500):
        super(SimpleDQNAgent, self).__init__(environment, memory_size)

        self.gamma = 0.8
        self.epsilon_decay = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.001

        self.current_model = self.build_model(self.furthest_distance)
        self.target_model = self.build_model(self.furthest_distance)
        self.update_target_model()
        self.annealer = SimpleDQNAnnealer(self, environment)

    def generate_random_action(self, protected_nodes):
        """
        Generates a random layer of swaps
        Care is taken to ensure that all swaps can occur in parallel
        That is, no two neighbouring edges undergo a swap simultaneously
        """

        action = np.array([0] * len(self.environment.edge_list))  # an action representing an empty layer of swaps

        edges = [(n1, n2) for (n1, n2) in self.environment.edge_list]
        edges = list(filter(lambda e: e[0] not in protected_nodes and e[1] not in protected_nodes, edges))
        edge_index_map = {edge: index for index, edge in enumerate(edges)}

        while len(edges) > 0:
            edge, action[edge_index_map[edge]] = random.sample(edges, 1)[0], 1
            edges = [e for e in edges if e[0] not in edge and e[1] not in edge]
        return action

    def obtain_target_nodes(self, current_state):  # TODO: rename
        """
        Obtains a vector that summarises the different distances
        from qubits to their targets.

        More precisely, x_i represents the number of qubits that are
        currently a distance of i away from their targets.

        If there are n qubits, then the length of this vector
        will also be n.
        """

        qubit_locations, qubit_targets, _, protected_nodes = current_state

        nodes_to_target_qubits = \
            [qubit_targets[qubit_locations[n]] for n in range(0, len(qubit_locations))]

        nodes_to_target_nodes = [next(iter(np.where(np.array(qubit_locations) == q)[0]), -1)
                                 for q in nodes_to_target_qubits]

        distance_vector = [0 for _ in range(self.furthest_distance)]

        for n in range(len(nodes_to_target_nodes)):
            target = nodes_to_target_nodes[n]

            if target == -1:
                continue

            d = int(self.environment.distance_matrix[n][target])
            distance_vector[d-1] += 1  # the vector is effectively indexed from 1

        return np.reshape(np.array(distance_vector), (1, self.furthest_distance))

    def act(self, current_state):
        """
        Chooses an action to perform in the environment and returns it
        (i.e. does not alter environment state)
        """

        protected_nodes = current_state[3]

        if np.random.rand() <= self.epsilon:
            action = self.generate_random_action(protected_nodes)
            return action, "Random"

        # Choose an action using the agent's current neural network
        action, _ = self.annealer.simulated_annealing(current_state, action_chooser='model')
        return action, "Model"

    def replay(self, batch_size):
        """
        Learns from past experiences
        """

        tree_index, minibatch, is_weights = self.memory_tree.sample(batch_size)
        minibatch_with_weights = zip(minibatch, is_weights)
        absolute_errors = []

        for experience, is_weight in minibatch_with_weights:
            [state, reward, next_state, done] = experience[0]

            target_nodes = self.obtain_target_nodes(state)
            next_target_nodes = self.obtain_target_nodes(next_state)

            q_val = self.current_model.predict(target_nodes)[0]

            if done:
                target = reward
            else:
                target = reward + self.gamma * self.target_model.predict(next_target_nodes)[0]

            absolute_error = abs(q_val - target)
            absolute_errors.append(absolute_error)

            self.current_model.fit(target_nodes, [target], epochs=1, verbose=0, sample_weight=is_weight)

        self.memory_tree.batch_update(tree_index, absolute_errors)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def used_up_memory_capacity(self):
        return self.memory_tree.tree.used_up_capacity
