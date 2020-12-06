"""
Annealer class for Single-State - TODO: Understand what single state is
"""

import random
import math
import numpy as np

import utils.static_heuristics as static_heuristics
import utils.action_edge_translation as action_edge_translation
from annealers.meta_annealer import MetaAnnealer


class Annealer(MetaAnnealer):

    def __init__(self, agent, environment):
        super().__init__(agent, environment)

    def check_valid_solution(self, solution, forced_mask):
        for i in range(len(solution)):
            if (forced_mask[i] == 1 and solution[i] == 0) or \
               (forced_mask[i] == -1 and solution[i] == 1):
                return False
        return super().check_valid_solution(solution, forced_mask)

    @staticmethod
    def acceptance_probability(current_energy, new_energy, temperature):
        if new_energy < current_energy:
            return 1
        else:
            energy_diff = new_energy - current_energy
            probability = math.exp(-energy_diff/temperature)
            return probability

    def get_energy(self, solution, current_state=None, action_chooser='model'):
        next_state_temp, _, _, _ = self.environment.step(solution, current_state)
        next_state_temp_nn_input = self.agent.obtain_target_nodes(next_state_temp)

        if action_chooser == 'model':
            q_val = self.agent.current_model.predict(next_state_temp_nn_input)[0]
        elif action_chooser == 'target':
            q_val = self.agent.target_model.predict(next_state_temp_nn_input)[0]
        else:
            raise ValueError('action_chooser can be either model or target, not %s' % (str(action_chooser)))

        return -q_val

    def generate_initial_solution(self, current_state):
        """
        Makes a random initial solution to start with by populating with whatever swaps possible

        :param current_state: State, the current state of mapping and progress
        :return: list, initial solution as boolean array of whether to swap each node
        """

        force_gates_action = static_heuristics.generate_force_gates_action(self.environment, current_state, version=3)

        if force_gates_action is None:
            num_edges = len(self.environment.edge_list)
            initial_solution = [0]*num_edges
            protected_mask = self.generate_protected_mask(current_state[3])

            available_edges = action_edge_translation.swappable_edges(
                initial_solution, current_state, protected_mask,
                self.environment.edge_list, self.environment.number_of_nodes)

            if not available_edges:
                return initial_solution, "None", protected_mask

            edge_index_to_swap = random.sample(available_edges, 1)[0]

            initial_solution[edge_index_to_swap] = (initial_solution[edge_index_to_swap] + 1) % 2

            return initial_solution, "Random", protected_mask
        else:
            return list(force_gates_action[0]), "Forced", force_gates_action[1]

    def generate_protected_mask(self, protected_nodes):
        """
        Make a list of edges which are blocked given nodes which are blocked

        :param protected_nodes: list, nodes that are being user elsewhere
        :return: list, edges that are blocked
        """
        return list(map(lambda e: -1 if e[0] in protected_nodes or e[1]
                                        in protected_nodes else 0, self.environment.edge_list))

    def simulated_annealing(self, current_state, action_chooser='model'):
        current_solution, method, forced_mask = self.generate_initial_solution(current_state)

        available_edges = action_edge_translation.swappable_edges(
            current_solution, current_state, forced_mask, self.environment.edge_list, self.environment.number_of_nodes)

        if not available_edges or current_solution == [0]*len(self.environment.edge_list):
            # There are no actions possible
            # Often happens when only one gate is left, and it's already been scheduled
            return current_solution, -np.inf

        return super()._simulated_annealing(current_solution, forced_mask,
                                            current_state, action_chooser, None)
