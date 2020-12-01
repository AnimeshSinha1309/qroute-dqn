import time as time_module
import random

from multiprocessing import Pool, cpu_count

from agents.paired_state_agent import DQNAgent
from environments.grid_environment import GridEnvironment
from agents.model_trainer import train_model
from agents.swap_scheduler import schedule_swaps
from utils.circuit_tools import generate_full_layer_circuit


def perform_run(n_rows, n_cols, training_episodes, test_episodes=100):
    """

    :param n_rows: dummy arg, number of rows of TODO???
    :param n_cols: dummy arg, number of rows of TODO???
    :param training_episodes: number of episodes to train for
    :param test_episodes: number of episodes to test for
    :return:
    """

    def circuit_generation_function(): return generate_full_layer_circuit(n_rows * n_cols).to_dqn_rep()

    environment = GridEnvironment(n_rows, n_cols, circuit_generation_function())
    agent = DQNAgent(environment)

    start_time = time_module.perf_counter()
    train_model(environment, agent, training_episodes=training_episodes,
                circuit_generation_function=circuit_generation_function, should_print=True)

    average_circuit_depth_overhead = 0.0

    for e in range(test_episodes):
        actions, circuit_depth = schedule_swaps(environment, agent,
                                                circuit=circuit_generation_function(), experience_db=None)
        average_circuit_depth_overhead += (1.0/test_episodes) * (circuit_depth - 1)
    end_time = time_module.perf_counter()

    time_taken = end_time - start_time
    datapoint = (n_rows, n_cols, average_circuit_depth_overhead, time_taken)
    print('Completed run:', datapoint)
    return datapoint


repeats = 5

grid_sizes = [(4, 4), (4, 5), (5, 5), (5, 6), (6, 6), (6, 7), (7, 7)]
inputs = [(4, 4, 50), (4, 5, 100), (5, 5, 120), (5, 6, 150), (6, 6, 200), (6, 7, 250), (7, 7, 300)] * repeats

random.shuffle(inputs)

if __name__ == '__main__':
    # PARAM: This sets up the multiprocessing, threshold on CPUs here so that I can test without it locally
    if cpu_count() > 8:
        p = Pool(cpu_count())
        results = p.starmap(perform_run, inputs)
    else:
        results = []
        for input_args in inputs:
            results.append(perform_run(*input_args))

    results.sort(key=lambda res: res[0] * res[1])
    print()
    for r in results:
        print(r)
    print()

    average_depth_overheads = {s: 0 for s in grid_sizes}

    for (n_rows, n_cols, depth_overhead, total_time) in results:
        average_depth_overheads[(n_rows, n_cols)] += (1.0 / repeats) * depth_overhead

    print(average_depth_overheads)
