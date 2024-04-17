import gym
from gym import spaces
import numpy as np
import pickle
import random
import matplotlib.pyplot as plt
import tensorflow as tf

def load_model(model_path):
    # Load a TensorFlow Keras model from a given path and confirm loading success.
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully.")
    return model

# Load model from specified path; ensure the path is correctly configured.
model = load_model('path/to/your/saved_model')

class AirfoilEnv(gym.Env):
    """ Custom Gym environment for airfoil optimization using Cl/Cd predictions from a neural network model. """
    
    def __init__(self, model):
        super(AirfoilEnv, self).__init__()
        self.model = model  # Surrogate model for Cl/Cd prediction
        # Define action space as continuous with specific ranges for each parameter modification.
        self.action_space = spaces.Box(low=np.array([-0.05, -0.1, -1]), high=np.array([0.05, 0.1, 1]), dtype=np.float32)
        # Define observation space that represents the range of possible airfoil parameters.
        self.observation_space = spaces.Box(low=np.array([0, 0, 5]), high=np.array([9, 9, 25]), dtype=np.float32)
        # Initialize environment state and benchmark Cl/Cd ratio.
        self.state, self.initial_cl_cd = self.reset_environment()
        self.intelligent_attempts = 0  # Track the number of intelligent attempts
        self.max_attempts = 8  # Maximum number of intelligent attempts
        self.e = 0  # Design requirement met flag

    def reset_environment(self):
        # Load airfoil data from a pickle file and randomly select an initial airfoil.
        with open('naca_airfoils.pkl', 'rb') as file:
            airfoils = pickle.load(file)
        initial_airfoil = random.choice(airfoils)
        # Extract parameters and initial Cl/Cd ratio from the selected airfoil.
        initial_params = initial_airfoil['params']
        initial_cl_cd = initial_airfoil.get('cl_cd', None)
        self.intelligent_attempts = 0
        self.e = 0
        return initial_params, initial_cl_cd

    def step(self, action):
        if self.e == 1 and self.intelligent_attempts < self.max_attempts:
            # Implement logic to choose a less optimal action if the design requirement is not met
            action = self.modify_action(action)
            self.intelligent_attempts += 1
            self.e = 0  # Reset the flag for the next attempt

        new_params = self.apply_action(self.state, action)
        new_cl_cd = self.predict_cl_cd(new_params)
        reward = self.compute_reward(self.initial_cl_cd, new_cl_cd)
        
        # Check if new parameters exceed the set range
        if not self.check_parameters(new_params):
            self.e = 1  # Design requirements are not met
            reward -= 10  # Penalize for exceeding parameter limits
        
        done = self.e == 1 and self.intelligent_attempts >= self.max_attempts
        self.state = new_params
        self.initial_cl_cd = new_cl_cd
        
        return self.state, reward, done, {}

    def check_parameters(self, params):
        """ Check if parameters are within the desired range. """
        return np.all(params >= np.array([0, 0, 5])) and np.all(params <= np.array([9, 9, 25]))

    def reset(self):
        # Reset environment to a new initial state.
        self.state, self.initial_cl_cd = self.reset_environment()
        return self.state

    def render(self):
        # Visualize the current airfoil configuration.
        self.plot_airfoil(self.state)

    def predict_cl_cd(self, parameters):
        # Predict Cl/Cd ratio using the surrogate model. Ensure parameter shape matches model expectations.
        parameters = parameters.reshape(1, -1)
        prediction = self.model.predict(parameters)
        cl, cd = prediction[0]
        return cl / cd if cd != 0 else 0

    def apply_action(self, current_params, action):
        # Apply the action to the current parameters and enforce constraints via clipping.
        new_params = current_params + action
        param_min_values = np.array([0, 0, 5])
        param_max_values = np.array([9, 9, 25])
        new_params = np.clip(new_params, param_min_values, param_max_values)
        return new_params

    def compute_reward(self, previous_cl_cd, current_cl_cd):
        # Reward is the improvement in Cl/Cd ratio.
        return current_cl_cd - previous_cl_cd

    def generate_airfoil_coordinates(self, naca_code):
        # Generate coordinates for plotting the airfoil based on its parameters.
        m, p, t = naca_code
        c = 1.0
        x = np.linspace(0, c, 100)
        yt = 5 * t * (0.2969 * np.sqrt(x/c) - 0.1260 * (x/c) - 0.3516 * (x/c)**2 + 0.2843 * (x/c)**3 - 0.1015 * (x/c)**4)
        y_upper = yt
        y_lower = -yt
        return x, y_upper, y_lower

    def plot_airfoil(self, params):
        # Plot the airfoil profile based on generated coordinates.
        x, y_upper, y_lower = self.generate_airfoil_coordinates(params)
        plt.figure(figsize=(10, 5))
        plt.plot(x, y_upper, 'b', label='Upper Surface')
        plt.plot(x, y_lower, 'r', label='Lower Surface')
        plt.fill_between(x, y_lower, y_upper, color='grey', alpha=0.3)
        plt.title(f'NACA Airfoil {params}')
        plt.xlabel('Chord Length')
        plt.ylabel('Thickness')
        plt.legend()
        plt.axis('equal')
        plt.grid(True)
        plt.show()

    def modify_action(self, action):
        """ Modify the action to be less optimal based on the previous failure. """
        # This could be implemented as reducing the magnitude of each action component
        # or selecting an alternative from a precomputed list of actions.
        return action * 0.9  # Simple example: reduce action magnitude by 10%