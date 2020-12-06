import time as time_module

from agents.paired_state_agent import DQNAgent
from environments.grid_environment import GridEnvironment
from utils.experience_db import ExperienceDB
from agents.model_trainer import train_model
from agents.swap_scheduler import schedule_swaps

start_time = time_module.clock()

experiment_name = "temp"
use_random_circuits = True
use_saved_model = False

n_rows = 4
n_cols = 4

if use_random_circuits:
    circuit = GridEnvironment.generate_random_circuit(n_rows * n_cols, n_rows * n_cols)
    def circuit_generation_function(): return GridEnvironment.generate_random_circuit(n_rows * n_cols, n_rows * n_cols)
else:
    circuit = []

    for i in range(n_rows * n_cols):
        if i % 2 == 0:
            circuit.append([i+1])
        else:
            circuit.append([i-1])

    if (n_rows * n_cols) % 2 == 1:
        circuit[-1] = []

    circuit_generation_function = None

print(circuit)

environment = GridEnvironment(n_rows, n_cols, circuit)
agent = DQNAgent(environment)
db = ExperienceDB()

training_episodes = 200 if n_rows * n_cols >= 36 else 100
test_episodes = 100

if __name__ == "__main__":

    if use_saved_model:
        agent.load_model(experiment_name)
    else:
        train_model(environment, agent, training_episodes=training_episodes,
                    circuit_generation_function=circuit_generation_function, should_print=True)
        agent.save_model(experiment_name)

    average_test_time = 0.0
    average_circuit_depth_overhead = 0.0

    for e in range(test_episodes):
        if circuit_generation_function is not None:
            circuit = circuit_generation_function()
        else:
            circuit = None

        actions, circuit_depth = schedule_swaps(environment, agent, circuit=circuit, experience_db=db)
        average_test_time += (1.0/test_episodes) * len(actions)
        average_circuit_depth_overhead += (1.0/test_episodes) * (circuit_depth - 1)

        print('Test time', len(actions))
        print('Total circuit depth', circuit_depth)
        print()

    print('Average time', average_test_time)
    print('Average depth overhead', average_circuit_depth_overhead)

    end_time = time_module.clock()
    total_time = end_time-start_time
    print('Total time taken:', total_time)
    db.write_to_disk(experiment_name)
