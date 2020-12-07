"""
Annealer class for Paired-State - TODO: Understand what paired state is
"""

import random
import math
import numpy as np
from collections import deque

import utils.action_edge_translation as action_edge_translation
from annealers.meta_annealer import MetaAnnealer


class DoubleDQNAnnealer(MetaAnnealer):

    def __init__(self, agent, environment):
        super().__init__(agent, environment)
        self.reversed_gates_deque = deque(maxlen=20)

    def check_valid_solution(self, solution, forced_mask):
        for i in range(len(solution)):
            if forced_mask[i] == 1 and solution[i] == 1:
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
        q_val = self.agent.get_quality(current_state, next_state_temp, action_chooser)
        return -q_val

    def generate_initial_solution(self, current_state, forced_mask):
        """
        Makes a random initial solution to start with by populating with whatever swaps possible

        :param current_state: State, the current state of mapping and progress
        :param forced_mask: list, mask of edges that are blocked
        :return: list, initial solution as boolean array of whether to swap each node
        """
        num_edges = len(self.environment.edge_list)
        initial_solution = [0]*num_edges

        available_edges = action_edge_translation.swappable_edges(
            initial_solution, current_state, forced_mask, self.environment.edge_list, self.environment.number_of_nodes)

        if not available_edges:
            return initial_solution

        edge_index_to_swap = random.sample(available_edges, 1)[0]

        initial_solution[edge_index_to_swap] = (initial_solution[edge_index_to_swap] + 1) % 2

        return initial_solution

    def generate_forced_mask(self, protected_nodes):
        """
        Make a list of edges which are blocked given nodes which are blocked

        :param protected_nodes: list, nodes that are being user elsewhere
        :return: list, edges that are blocked
        """
        return list(map(lambda e: True if e[0] in protected_nodes or
                                          e[1] in protected_nodes else False, self.environment.edge_list))

    @staticmethod
    def calculate_reversed_gates_proportion(suggestion, solution):
        """
        Calculates percentage of gates that are suggested but not really swapped in the solution

        :param suggestion: boolean array of gates that are suggested for swaps
        :param solution: boolean array of gates that are actually in the final solution
        :return: fraction of gates in suggestion not in solution
        """
        reversed_gates = [suggestion[i] == 1 and solution[i] == 0 for i in range(len(suggestion))]

        if sum(suggestion) == 0 or sum(reversed_gates) == 0:
            return 0.0

        return float(sum(reversed_gates)) / float(sum(suggestion))

    def simulated_annealing(self, current_state, action_chooser='model', search_limit=None):
        forced_mask = self.generate_forced_mask(current_state.protected_nodes)
        current_solution = self.generate_initial_solution(current_state, forced_mask)

        if current_solution == [0] * len(self.environment.edge_list):
            # There are no actions possible often happens when only one gate is left, and it's already been scheduled
            if action_chooser == 'model':
                return current_solution, np.array([-np.inf])
            else:
                return current_solution, np.array([0])

        return super()._simulated_annealing(current_solution, forced_mask,
                                            current_state, action_chooser, search_limit)
