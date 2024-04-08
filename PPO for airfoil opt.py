import gym
from gym import spaces
import numpy as np
import pickle
import random
import matplotlib.pyplot as plt
import tensorflow as tf

# Assuming the necessary imports and model loading happen here
# Path to the saved model
model_path = 'path/to/your/saved_model'

# Load the model
model = tf.keras.models.load_model(model_path)

print("Model loaded successfully.")


class NACA4ActionSpace:
    def __init__(self, dm_range=(-0.05, 0.05), dp_range=(-0.1, 0.1), dt_range=(-1, 1)):
        self.dm_range = dm_range
        self.dp_range = dp_range
        self.dt_range = dt_range

    def sample(self):
        delta_m = np.random.uniform(self.dm_range[0], self.dm_range[1])
        delta_p = np.random.uniform(self.dp_range[0], self.dp_range[1])
        delta_t = np.random.uniform(self.dt_range[0], self.dt_range[1])
        return np.array([delta_m, delta_p, delta_t])

class AirfoilEnv(gym.Env):
    def __init__(self, model):
        super(AirfoilEnv, self).__init__()
        self.model = model  # Surrogate model for Cl/Cd prediction
        self.action_space = spaces.Box(low=np.array([-0.05, -0.1, -1]), high=np.array([0.05, 0.1, 1]), dtype=np.float32)
        self.observation_space = spaces.Box(low=np.array([0, 0, 5]), high=np.array([9, 9, 25]), dtype=np.float32)
        self.state, self.initial_cl_cd = self.reset_environment()
        
    def step(self, action):
        new_params = apply_action(self.state, action)
        new_cl_cd = predict_cl_cd(new_params, self.model)
        reward = compute_reward(self.initial_cl_cd, new_cl_cd)
        self.state = new_params
        self.initial_cl_cd = new_cl_cd  # Update for the next step's reward calculation
        done = False  # Define your own termination condition
        return self.state, reward, done, {}
        
    def reset(self):
        self.state, self.initial_cl_cd = self.reset_environment()
        return self.state

    def render(self):
        plot_airfoil(self.state)

    # Add other necessary methods here

def predict_cl_cd(parameters, model):
    parameters = parameters.reshape(1, -1)
    prediction = model.predict(parameters)
    cl, cd = prediction[0]
    return cl / cd if cd != 0 else 0

def apply_action(current_params, action):
    new_params = current_params + action
    # Define min and max values based on your constraints for m, p, and t
    param_min_values = np.array([0, 0, 5])
    param_max_values = np.array([9, 9, 25])
    new_params = np.clip(new_params, param_min_values, param_max_values)
    return new_params

def compute_reward(previous_cl_cd, current_cl_cd):
    return current_cl_cd - previous_cl_cd

def reset_environment():
    # Your existing reset_environment function remains unchanged
    with open('naca_airfoils.pkl', 'rb') as file:
        airfoils = pickle.load(file)
    initial_airfoil = random.choice(airfoils)
    
    # Assuming each airfoil entry contains parameters and possibly a starting Cl/Cd
    initial_params = initial_airfoil['params']
    initial_cl_cd = initial_airfoil.get('cl_cd', None)
    
    # Initialize environment state with selected airfoil
    # You may need to adjust this part based on how your environment and state are structured
    state = initial_params
    return state, initial_cl_cd

def generate_airfoil_coordinates(naca_code):
    m = int(naca_code[0]) / 100.0  # Maximum camber
    p = int(naca_code[1]) / 10.0   # Position of maximum camber
    t = int(naca_code[2:]) / 100.0  # Thickness

    c = 1.0  # Chord length, assumed to be 1 for simplicity
    x = np.linspace(0, c, 100)  # x-coordinates along the chord

    # Thickness distribution for symmetrical airfoil
    yt = 5 * t * (0.2969 * np.sqrt(x/c) - 0.1260 * (x/c) - 0.3516 * (x/c)**2 + 0.2843 * (x/c)**3 - 0.1015 * (x/c)**4)

    # Assuming a symmetrical airfoil for simplicity (no camber line calculation)
    y_upper = yt
    y_lower = -yt

    return x, y_upper, y_lower

def plot_airfoil(params):
    x, y_upper, y_lower = generate_airfoil_coordinates(params)
    
    plt.figure(figsize=(10, 5))
    plt.plot(x, y_upper, 'b', label='Upper Surface')
    plt.plot(x, y_lower, 'r', label='Lower Surface')
    plt.fill_between(x, y_lower, y_upper, color='grey', alpha=0.3)
    plt.title(f'NACA {params} Airfoil')
    plt.xlabel('Chord Length')
    plt.ylabel('Thickness')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.show()