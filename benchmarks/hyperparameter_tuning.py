import numpy as np
import time as time_module

from multiprocessing import Pool, cpu_count

from agents.single_state_agent import SimpleDQNAgent
from environments.grid_environment import GridEnvironment
from agents.model_trainer import train_model
from agents.swap_scheduler import schedule_swaps

use_random_circuits = False

n_rows = 5
n_cols = 5

if use_random_circuits:
    circuit = GridEnvironment.generate_random_circuit(n_rows * n_cols, n_rows * n_cols)
else:
    circuit = []

    for i in range(n_rows * n_cols):
        if i % 2 == 0:
            circuit.append([i+1])
        else:
            circuit.append([i-1])

if (n_rows * n_cols) % 2 == 1:
    circuit[-1] = []

print(circuit)

training_episodes = 400 if n_rows * n_cols >= 36 else 200
test_episodes = 100


def test_parameters(_gamma):

    start_time = time_module.clock()

    environment = GridEnvironment(n_rows, n_cols, circuit)
    agent = SimpleDQNAgent(environment)

    train_model(environment, agent, training_episodes=training_episodes, should_print=False)

    average_test_time = 0.0
    average_circuit_depth_overhead = 0.0

    for e in range(test_episodes):
        actions, circuit_depth = schedule_swaps(environment, agent, experience_db=None)
        average_test_time += (1.0/test_episodes) * len(actions)
        average_circuit_depth_overhead += (1.0/test_episodes) * (circuit_depth - 1)

    end_time = time_module.clock()

    total_time = end_time-start_time

    return average_circuit_depth_overhead, total_time


def perform_run(params):
    print("Starting run: " + str(params))

    performance_data = []
    runtime_data = []

    for _ in range(10):
        average_test_time, total_time = test_parameters(params)

        performance_data.append(average_test_time)
        runtime_data.append(total_time)

        print('Completed run:', (params, average_test_time))

    datapoint = (params, np.mean(performance_data), np.mean(runtime_data))

    return datapoint


inputs = []

for gamma in np.linspace(0.2, 0.9, 8):
    inputs.append(gamma)

np.random.shuffle(inputs)

if __name__ == '__main__':
    p = Pool(cpu_count())
    print(p.map(perform_run, inputs))
