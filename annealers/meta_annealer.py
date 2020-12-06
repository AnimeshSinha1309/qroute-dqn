"""
Abstract Annealer module, implements simulated annealing
"""

import random
import copy
import numpy as np

import utils.action_edge_translation as action_edge_translation


class MetaAnnealer:
    """
    Class to perform simulated annealing using a value function approximator
    """

    def __init__(self, agent, environment):
        """
        Sets hyper-parameters and stores the agent and environment to initialize Annealer

        :param agent: Agent, to evaluate the value function
        :param environment: environment, maintaining the device and state
        """
        self.initial_temperature = 60.0
        self.min_temperature = 0.1
        self.cooling_multiplier = 0.95
        self.environment = environment
        self.agent = agent

        self.safety_checks_on = True
        self.speed_over_optimality = False

    def get_neighbour_solution(self, current_solution, current_state, forced_mask):
        """
        Get a solution neighboring current, that is one swap inserted
        :param current_solution: list of edges to swap, current solution to start with
        :param current_state: State, the current state of mapping and progress
        :param forced_mask: list, which edges cannot be swapped
        :return: list, neighbor solution
        """
        neighbour_solution = copy.copy(current_solution)
        edge_list = self.environment.edge_list
        n_nodes = self.environment.number_of_nodes

        available_edges = action_edge_translation.swappable_edges(neighbour_solution, current_state,
                                                                  forced_mask, edge_list, n_nodes)

        if not available_edges:
            exit("Ran out of edges to swap")

        edge_index_to_swap = random.sample(available_edges, 1)[0]

        neighbour_solution[edge_index_to_swap] = (neighbour_solution[edge_index_to_swap] + 1) % 2

        if self.safety_checks_on and not self.check_valid_solution(neighbour_solution, forced_mask):
            exit("Solution not safe")

        return neighbour_solution

    def get_energy(self, solution, current_state=None, action_chooser='model'):
        """
        Returns the energy function (negative value function) for the current state using the model.
        :param solution: list of edges to swap as a boolean array
        :param current_state: State, the state at the current moment (q_locations, q_targets, protected_nodes, ...)
        :param action_chooser: str, if model, the current model is used to compute the value function,
                                    if target, then the target model is used.
        :return: int or float, the energy value
        """
        raise NotImplementedError('Meta-Annealer should not be instantiated and called to get_energy')

    @staticmethod
    def acceptance_probability(current_energy, new_energy, temperature):
        """
        Compute acceptance probability given delta-energy

        :param current_energy: int/float, initial energy (negative of value function)
        :param new_energy: int/float, final energy (negative of value function)
        :param temperature: int/float, temperature in the simulation (randomness)
        :return: int or float, probability to accept
        """
        raise NotImplementedError('Meta-Annealer should not be instantiated and called to acceptance_probability')

    def check_valid_solution(self, solution, forced_mask):
        """
        Checks if a solution is valid, i.e. does not use one node twice

        :param solution: list, boolean array of swaps, the solution to check
        :param forced_mask: list, blocking swaps which are not possible
        :return: True if valid, False otherwise
        """
        if 1 in solution:
            swap_edge_indices = np.where(np.array(solution) == 1)[0]
            swap_edges = [self.environment.edge_list[index] for index in swap_edge_indices]
            swap_nodes = [node for edge in swap_edges for node in edge]

            # return False if repeated swap nodes
            seen = set()
            for node in swap_nodes:
                if node in seen:
                    return False
                seen.add(node)
            return True

        return True  # TODO should all zero be valid action?

    def _simulated_annealing(self, current_solution, forced_mask,
                             current_state, action_chooser='model', search_limit=None):
        """
        Uses Simulated Annealing to find the next best state based on combinatorial
        actions taken by the agent.

        :param current_solution: list of len(device_topology), whether each edge is being flipped or not
        :param forced_mask:
        :param current_state: State, the state before this iterations of sim-anneal
        :param action_chooser: str, if model, uses the model for value function
        :param search_limit:
        :return: best_solution, value of best_energy
        """
        temp = self.initial_temperature
        current_energy = self.get_energy(current_solution, current_state=current_state, action_chooser=action_chooser)
        best_solution = copy.copy(current_solution)
        best_energy = current_energy

        iterations_since_best = 0
        iterations = 0

        while temp > self.min_temperature:
            if self.speed_over_optimality and iterations_since_best > 40:
                break
            elif search_limit is not None and iterations > search_limit:
                break

            new_solution = self.get_neighbour_solution(current_solution, current_state, forced_mask)
            new_energy = self.get_energy(new_solution, current_state=current_state, action_chooser=action_chooser)
            accept_prob = self.acceptance_probability(current_energy, new_energy, temp)

            if accept_prob > random.random():
                current_solution = new_solution
                current_energy = new_energy

                # Save best solution, so it can be returned if algorithm terminates at a sub-optimal solution
                if current_energy < best_energy:
                    best_solution = copy.copy(current_solution)
                    best_energy = current_energy
                    # intervals.append(iterations_since_best)
                    iterations_since_best = 0

            temp = temp * self.cooling_multiplier
            iterations_since_best += 1
            iterations += 1

        return best_solution, best_energy
