import abc

import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import model_from_json

from annealers.single_state_annealer import Annealer
from utils.PER_memory_tree import Memory


class MetaDQNAgent(abc.ABC):

    def __init__(self, environment, memory_size):
        self.environment = environment
        self.furthest_distance = int(np.amax(self.environment.distance_matrix))
        self.max_node_degree = int(np.max(np.sum(self.environment.adjacency_matrix, axis=1)))
        self.memory_size = memory_size

        self.gamma = None
        self.epsilon_decay = None
        self.epsilon = None
        self.epsilon_min = None
        self.learning_rate = 0.001

        self.current_model = None
        self.target_model = None

        self.memory_tree = Memory(memory_size)
        self.annealer = Annealer(self, environment)

    def build_model(self, input_size):
        """
        Build the neural network model for this agent
        """
        model = Sequential([
            Dense(32, input_dim=input_size, activation='relu'),
            Dense(32, activation='relu'),
            Dense(32, activation='relu'),
            Dense(1, activation='linear'),
        ])
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        """
        Copy weights from the current model to the target model
        """
        self.target_model.set_weights(self.current_model.get_weights())

    def save_model(self, model_name=None):
        # Serialize model to JSON
        model_json = self.current_model.to_json()

        if model_name is not None:
            filepath = "./models/" + model_name
        else:
            filepath = "./models/agent_model"

        with open(filepath + ".json", "w") as json_file:
            json_file.write(model_json)

        # Serialize weights to HDF5
        self.current_model.save_weights(filepath + ".h5")
        print("Saved model to disk")

    def remember(self, state, reward, next_state, done):
        """
        Store experience in the memory tree
        """
        self.memory_tree.store((state, reward, next_state, done))

    def load_model(self, model_name=None):
        self.epsilon = self.epsilon_min

        if model_name is not None:
            filepath = "./models/" + model_name
        else:
            filepath = "./models/agent_model"

        # Load json and create model
        json_file = open(filepath + '.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.current_model = model_from_json(loaded_model_json)

        # Load weights into new model
        self.current_model.load_weights(filepath + ".h5")
        self.current_model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        self.update_target_model()
        print("Loaded model from disk")
