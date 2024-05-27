import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from reservoir_model import RecurrentNetworkLQR2
from train_reservoir import train_reservoir_model
from lorenz_attractor import LorenzAttractorDataGenerator
import threading
import csv

# Define your network configuration
base_config = {
    'Ni': 1,  # Number of input neurons. Determines how many different input signals the network can handle.
    'N': 250,  # Total number of neurons in the main recurrent layer (reservoir).
    'N_u': 250,  # Total number of neurons in the feedback reservoir.
    'No': 1,  # Total number of output neurons. Defines the dimensionality of the network's output.
    'No_u': 1,  # Total number of output neurons in the feedback output layer.
    'tau': 10,  # Time constant for the neurons in the main reservoir. Influences the decay rate of neuronal activation.
    'tau_u': 10,  # Time constant for the neurons in the feedback input layer. Affects how quickly these neurons react to changes.
    'g': 1.5,  # Scaling factor for the synaptic weights in the main reservoir. Influences the overall dynamics and stability of the reservoir.
    'g_u': 1.5,  # Scaling factor for the synaptic weights in the feedback input layer. Controls the impact of feedback inputs on the network.
    'pc': 0.2,  # Connectivity probability within the main reservoir. Determines the sparsity of the reservoir connections.
    'pc_u': 0.2,  # Connectivity probability within the feedback input layer. Affects the sparsity of connections in the feedback network.
    'Io': 1.0,  # Intensity of the noise added to the input. Can be used to simulate real-world variability and improve robustness.
    'P_plastic': 0.2,  # Proportion of plastic (modifiable) synapses in the main reservoir. A value of 1.0 means all synapses are plastic.
    'P_plastic_u': 0.2  # Proportion of plastic synapses in the feedback input layer. Determines how adaptable the feedback inputs are.
}

feedback = False
random_feedback = False
Noise = True

# Noise ranging from 0.001, 0.01, 0.1 to 1.0
experiment_name = "experiment2_Io1"

# Load the saved data for experiments
data_generator = LorenzAttractorDataGenerator()
lorenz_data = data_generator.load_data('lorenz_data1.npy')

# Print the loaded data
data_generator.print_data(lorenz_data)

# Initialize the network
model_force = RecurrentNetworkLQR2(base_config)
model_lqr = RecurrentNetworkLQR2(base_config)

trials = 100
print(experiment_name)

def train_and_save_data(trials, model, lorenz_data, filename, use_lqr):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(['Iteration', 'Prediction', 'Target', 'Error', 'Total Error', 'Average Error', 'MSE'])
        
        print(f'Start Training with {"LQR" if use_lqr else "FORCE"}')
        for i in range(trials):
            print(f'Training {"LQR" if use_lqr else "FORCE"}: Trial {i} of {trials}')
            predictions, targets, errors, total_error, avg_error, mse = train_reservoir_model(model, lorenz_data, apply_noise=Noise, apply_feedback=feedback, random_feedback=random_feedback, use_lqr=use_lqr)
            # Write data for each trial
            for pred, target, error in zip(predictions, targets, errors):
                writer.writerow([i, pred, target, error, total_error, avg_error, mse])   
                
        print(f'Data saved to {filename}')
        
# Create shared lists to store average errors
avg_error_lqr = []
avg_error_no_lqr = []

# Create threads for each training loop
thread_lqr = threading.Thread(target=train_and_save_data, args=(trials, model_lqr, lorenz_data, f'{experiment_name}_lqr_data.csv', True))
thread_force = threading.Thread(target=train_and_save_data, args=(trials, model_force, lorenz_data, f'{experiment_name}_force_data.csv', False))

# Start the threads
thread_lqr.start()
thread_force.start()

# Wait for both threads to finish
thread_lqr.join()
thread_force.join()

# Read data from CSV files
df_lqr = pd.read_csv(f'{experiment_name}_lqr_data.csv')
df_force = pd.read_csv(f'{experiment_name}_force_data.csv')

# Extract average errors for each method
avg_error_lqr = df_lqr['Average Error']
avg_error_no_lqr = df_force['Average Error']

# Plot the average errors
plt.figure(figsize=(8, 6))
plt.plot(avg_error_lqr, label='LQR')
plt.plot(avg_error_no_lqr, label='FORCE')
plt.xlabel('Iteration')
plt.ylabel('Average Error')
plt.title('Average Error: LQR vs FORCE')
plt.legend()
plt.show()

# Flatten predictions and targets
predictions_flat_lqr = np.ravel(df_lqr['Prediction'])
targets_flat_lqr = np.ravel(df_lqr['Target'])

predictions_flat_no_lqr = np.ravel(df_force['Prediction'])
targets_flat_no_lqr = np.ravel(df_force['Target'])

# Plot predictions and flattened targets for both cases
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(predictions_flat_lqr, label='Predictions (with LQR)')
plt.plot(predictions_flat_no_lqr, label='Predictions (with FORCE)')
plt.plot(targets_flat_no_lqr, label='Targets')
plt.xlabel('Time Step')
plt.ylabel('Lorenz Data')
plt.title('Lorenz Data vs. Target')
plt.legend()

# Plot absolute distance between predictions and targets
plt.subplot(1, 2, 2)
plt.plot(np.abs(predictions_flat_lqr - targets_flat_lqr), label='Absolute Distance (with LQR)')
plt.plot(np.abs(predictions_flat_no_lqr - targets_flat_no_lqr), label='Absolute Distance (with FORCE)')
plt.xlabel('Time Step')
plt.ylabel('Absolute Distance')
plt.title('Absolute Distance between Predictions and Targets')
plt.legend()

plt.tight_layout()
plt.show()
