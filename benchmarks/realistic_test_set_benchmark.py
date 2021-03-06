import time as time_module
import random

from multiprocessing import Pool, cpu_count

from agents.paired_state_agent import DoubleDQNAgent
from environments.grid_environment import GridEnvironment
from agents.model_trainer import train_model
from agents.swap_scheduler import schedule_swaps
from utils.circuit_tools import generate_completely_random_circuit
from utils.realistic_test_set_tools import import_test_set

training_episodes = 100
should_train = True

test_set_circuits = import_test_set()


def train_model_on_random_circuits(f_model_number):
    model_name = "random_circuits_" + str(f_model_number)

    def training_circuit_generation_function(): return generate_completely_random_circuit(16, 50).to_dqn_rep()

    environment = GridEnvironment(4, 4, training_circuit_generation_function())
    agent = DoubleDQNAgent(environment)

    train_model(environment, agent, training_episodes=training_episodes,
                circuit_generation_function=training_circuit_generation_function, should_print=False)
    agent.save_model(model_name)


def perform_run(f_initial_locations, f_model_number):
    model_name = "random_circuits_" + str(f_model_number)

    start_time = time_module.clock()

    environment = GridEnvironment(4, 4, test_set_circuits[0].to_dqn_rep())
    agent = DoubleDQNAgent(environment)
    agent.load_model(model_name)

    average_test_time = 0.0
    average_circuit_depth_overhead = 0.0
    average_circuit_depth_ratio = 0.0

    test_episodes = len(test_set_circuits)

    for e in range(test_episodes):
        circuit = test_set_circuits[e]
        f_qubit_locations = f_initial_locations[e]
        original_depth = circuit.depth()

        actions, circuit_depth = schedule_swaps(environment, agent, circuit=circuit, experience_db=None,
                                                qubit_locations=f_qubit_locations, safety_checks_on=True)
        average_test_time += (1.0/test_episodes) * len(actions)
        average_circuit_depth_overhead += (1.0/test_episodes) * (circuit_depth - original_depth)
        average_circuit_depth_ratio += (1.0/test_episodes) * (float(circuit_depth)/float(original_depth))

    end_time = time_module.clock()

    time_taken = end_time - start_time
    result = (f_model_number, average_test_time, average_circuit_depth_overhead,
              average_circuit_depth_ratio, time_taken)

    print('Completed run:', result)

    return result


repeats = 5

random.seed(343)

initial_locations_sets = []

for _ in range(repeats):
    initial_locations = []

    for _ in range(len(test_set_circuits)):
        qubit_locations = list(range(16))
        random.shuffle(qubit_locations)
        initial_locations.append(qubit_locations)

    initial_locations_sets.append(initial_locations)

inputs = []

for i in range(repeats):
    for loc in initial_locations_sets:
        inputs.append((loc, i))

random.shuffle(inputs)

if __name__ == '__main__':
    p = Pool(cpu_count())

    if should_train:
        model_numbers = list(range(0, repeats))
        p.map(train_model_on_random_circuits, model_numbers)

    results = p.starmap(perform_run, inputs)

    results.sort(key=lambda res: res[0])

    print()
    for r in results:
        print(r)
    print()

    average_depth_ratios = {k: 0 for k in list(range(repeats))}

    for (model_number, test_time, depth_overhead, depth_ratio, total_time) in results:
        average_depth_ratios[model_number] += (1.0/repeats) * depth_ratio

    print(average_depth_ratios)
