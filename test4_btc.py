import numpy as np
import matplotlib.pyplot as plt
import control
import pandas as pd
import math

class RecurrentNetworkLQR2(object):
    def __init__(self, config):
        # Copy the parameters
        self.Ni = config['Ni']
        self.N = config['N']
        self.N_u = config['N_u']
        self.No = config['No']
        self.No_u = config['No_u']
        self.tau = config['tau']
        self.tau_u = config['tau_u']
        self.g = config['g']
        self.g_u = config['g_u']
        self.pc = config['pc']
        self.pc_u = config['pc_u']
        self.Io = config['Io']
        self.P_plastic = config['P_plastic']
        self.P_plastic_u = config['P_plastic_u']
        self.N_plastic = math.floor(self.P_plastic*self.N)
        self.N_plastic_u = math.floor(self.P_plastic_u*self.N_u)
        
        # Input
        self.I = np.zeros((self.Ni, 1))

        # Recurrent population
        self.x = np.random.uniform(-1.0, 1.0, (self.N, 1))
        self.x_u = np.random.uniform(-1.0, 1.0, (self.N_u, 1))
        self.r_x = np.tanh(self.x)
        self.r_u = np.tanh(self.x_u)

        # Read-out population
        self.y = np.zeros((self.No, 1))
        self.u = np.zeros((self.No_u, 1))

        # Weights between the input and recurrent units
        self.W_in = np.random.randn(self.N, self.Ni)

        # Weights between the recurrent units
        self.W_rec = (np.random.randn(self.N, self.N) * self.g/np.sqrt(self.pc*self.N))
        self.W_rec_u = (np.random.randn(self.N_u, self.N_u) * self.g_u/np.sqrt(self.pc_u*self.N_u))

        # The connection pattern is sparse with p=0.1
        connectivity_mask = np.random.binomial(1, self.pc, (self.N, self.N))
        connectivity_mask[np.diag_indices(self.N)] = 0
        self.W_rec *= connectivity_mask

        connectivity_mask_u = np.random.binomial(1, self.pc_u, (self.N_u, self.N_u))
        connectivity_mask_u[np.diag_indices(self.N_u)] = 0
        self.W_rec_u *= connectivity_mask_u

        # Store the pre-synaptic neurons to each plastic neuron
        self.W_plastic = [list(np.nonzero(connectivity_mask[i, :])[0]) for i in range(self.N_plastic)]
        self.W_plastic_u = [list(np.nonzero(connectivity_mask_u[i, :])[0]) for i in range(self.N_plastic_u)]

        # Output weights
        self.W_out = (np.random.randn(self.No, self.N) / np.sqrt(self.N))
        self.W_out_u = (np.random.randn(self.No_u, self.N_u) / np.sqrt(self.N_u))

        # Inverse correlation matrix of inputs for learning recurrent weights
        self.P = [np.identity(len(self.W_plastic[i])) for i in range(self.N_plastic)]
        self.P_u = [np.identity(len(self.W_plastic_u[i])) for i in range(self.N_plastic_u)]

        # Inverse correlation matrix of inputs for learning readout weights
        self.P_out = [np.identity(self.N) for i in range(self.No)]
        self.P_out_u = [np.identity(self.N_u) for i in range(self.No_u)]

        # Weights for interactions between reservoir and thalamus neurons
        # Weight matrix from thalamus to reservoir
        self.W_cross = np.random.randn(self.N, self.N_u) * 1.0 / np.sqrt(self.N_u)
        # Weight matrix from reservoir to thalamus
        self.W_cross_u = np.random.randn(self.N_u, self.N) * 1.0 / np.sqrt(self.N)

    def step(self, I, noise, feedback):
        """
        Updates neural variables for a single simulation step.
        
        * `h`: input at time t, numpy array of shape (Ni, 1)
        * `h_const`: constant input noise
        """

        # Noise can be shut off
        I_noise = (self.Io * np.random.randn(self.N, 1) if noise else 0.0)
        
        # tau * dx/dt + x = I + W_rec * r + I_noise
        
        if feedback == False:
            self.x += (np.dot(self.W_in, I) + np.dot(self.W_rec, self.r_x) + I_noise - self.x)/self.tau
        else:
            self.x += (np.dot(self.W_in, I) + np.dot(self.W_rec, self.r_x) + self.r_u + I_noise - self.x)/self.tau
            self.x_u += (np.dot(self.W_rec_u, self.r_u) + self.r_x - self.x_u) / self.tau_u

        # r = tanh(x)
        self.r_x = np.tanh(self.x)
        self.r_u = np.tanh(self.x_u)
        
        # z = W_out * r
        self.y = np.dot(self.W_out, self.r_x)
        self.u = np.dot(self.W_out_u, self.r_u)

    def train_recurrent(self, target):
            """
            Applies the RLS learning rule to the recurrent weights.
            
            * `target`: desired trajectory at time t, numpy array of shape (N, 1)
            """
            # Compute the error of the recurrent neurons
            error_x = self.r_x - target

            # Apply the FORCE learning rule to the recurrent weights
            for i in range(self.N_plastic): # for each plastic post neuron
                
                # Get the rates from the plastic synapses only
                r_plastic = self.r_x[self.W_plastic[i]]
                
                # Multiply the inverse correlation matrix P*R with the rates from the plastic synapses only
                PxR = np.dot(self.P[i], self.r_x[self.W_plastic[i]])
                
                # Normalization term 1 + R'*P*R
                RxPxR = (1. + np.dot(r_plastic.T,  PxR))
                
                # Update the inverse correlation matrix P <- P - ((P*R)*(P*R)')/(1+R'*P*R)
                self.P[i] -= np.dot(PxR, PxR.T)/RxPxR
                
                # Learning rule W <- W - e * (P*R)/(1+R'*P*R)
                self.W_rec[i, self.W_plastic[i]] -= error_x[i, 0] * (PxR/RxPxR)[:, 0]
            return np.abs(error_x).sum() / self.N_plastic
        
    def train_recurrent_lqr(self, target, Q_val, R_val):
            """
            Applies an LQR-like learning rule to the recurrent weights based on a defined cost function.

            * `target`: desired trajectory at time t, numpy array of shape (N, 1)
            * `Q_val`: scalar representing the cost of deviation from the desired neuron activity
            * `R_val`: scalar representing the cost of changing weights (control input)
            """
            # Define the 'A' matrix (system dynamics), decay towards zero without input
            A = np.eye(self.N_plastic) * -1

            # Define the 'B' matrix (control dynamics), unitary effect of weight changes
            B = np.eye(self.N_plastic)

            # Define the 'Q' matrix (cost of state deviation)
            Q = np.eye(self.N_plastic) * Q_val

            # Define the 'R' matrix (cost of control effort)
            R = np.eye(self.N_plastic) * R_val

            # Compute the LQR gain 'K'
            K, _, _ = control.lqr(A, B, Q, R)

            # Initialize total error for monitoring
            total_error = 0

            # Compute the error of the recurrent neurons (this is analogous to the state deviation in LQR)
            error_x = self.r_x - target
            
            # Print shapes:
            #print("Shape of A:", A.shape)
            #print("Shape of B:", B.shape)
            #print("Shape of K:", K.shape)
            #print("Shape of error_x:", error_x.shape)
            #print("Shape of a specific error_x (0):", error_x[0].shape)
            #print(self.N_plastic)
            # Apply the LQR-like update rule to the recurrent weights
            for i in range(self.N_plastic):  # for each plastic post neuron
                # Apply LQR control law: change in weights = -K * error
                delta_W = -np.dot(K, error_x[0:self.N_plastic])
                    
                #print(self.W_plastic[i])
                #print("Shape of delta_W:", delta_W.shape)
                    
                # Update the recurrent weights based on the error and 'gain'
                # Apply adjustments for connected neurons only
                self.W_rec[i, self.W_plastic[i]] += delta_W[i, 0]  # Adjust specific weights

                # Accumulate total error
                total_error += np.abs(error_x[i, 0])

            # Return the average error for monitoring
            return np.abs(error_x).sum() / self.N_plastic


    def train_recurrent_thalamus(self, target):
        """
        Applies the RLS learning rule to the recurrent weights.

        * `target`: desired trajectory at time t, numpy array of shape (N, 1)
        """
        # Compute the error of the recurrent neurons
        error_u = self.r_u - target

        # Apply the FORCE learning rule to the recurrent weights
        for i in range(self.N_plastic_u): # for each plastic post neuron

            # Get the rates from the plastic synapses only
            r_plastic = self.r_u[self.W_plastic_u[i]]

            # Multiply the inverse correlation matrix P*R with the rates from the plastic synapses only
            PxR = np.dot(self.P_u[i], self.r_u[self.W_plastic_u[i]])

            # Normalization term 1 + R'*P*R
            RxPxR = (1. + np.dot(r_plastic.T,  PxR))

            # Update the inverse correlation matrix P <- P - ((P*R)*(P*R)')/(1+R'*P*R)
            self.P_u[i] -= np.dot(PxR, PxR.T)/RxPxR

            # Learning rule W <- W - e * (P*R)/(1+R'*P*R)
            self.W_rec_u[i, self.W_plastic_u[i]] -= error_u[i, 0] * (PxR/RxPxR)[:, 0]
            
        # Return the average error for monitoring
        return np.abs(error_u).sum() / self.N_plastic_u

    def train_readout(self, target):
        """
        Applies the RLS learning rule to the readout weights.

        * `target`: desired output at time t, numpy array of shape (No, 1)
        """

        # Compute the error of the output neurons
        error_y = self.y - target

        # Apply the FORCE learning rule to the readout weights
        for i in range(self.No): # for each readout neuron

            # Multiply the rates with the inverse correlation matrix P*R
            PxR = np.dot(self.P_out[i], self.r_x)

            # Normalization term 1 + R'*P*R
            RxPxR = (1. + np.dot(self.r_x.T,  PxR))

            # Update the inverse correlation matrix P <- P - ((P*R)*(P*R)')/(1+R'*P*R)
            self.P_out[i] -= np.dot(PxR, PxR.T)/RxPxR

            # Learning rule W <- W - e * (P*R)/(1+R'*P*R)
            self.W_out[i, :] -= error_y[i, 0] * (PxR/RxPxR)[:, 0]
            
        # Return the average error for monitoring
        return np.abs(error_y).sum() / self.No

def train_reservoir_model(rn, data, apply_noise=True, apply_feedback=False, use_lqr=False, print_step_error=False):
    """
    Train a reservoir model with customizable parameters.

    Args:
    - rn: The initialized reservoir model object.
    - data: The input data (time series) for training.
    - apply_noise: Boolean indicating whether to apply noise during training.
    - apply_feedback: Boolean indicating whether to apply feedback during training.
    - use_lqr: Boolean indicating whether to use LQR for training.
    - print_step_error: Boolean indicating whether to print the error at each step.

    Returns:
    - predictions: List containing the predictions made by the model.
    - targets: List containing the target values corresponding to the predictions.
    - total_error: Total error accumulated during training.
    """

    # Initialize lists to store predictions and targets
    predictions = []
    targets = []

    # Run the network and apply LQR control to follow the input time series
    max_steps = len(data) - 1
    total_error = 0  # For performance evaluation
    for step in range(max_steps):
        # Get the current target state from the input data
        target_state = data[step+1]
        current_state = data[step]

        # Update network and apply control as usual
        rn.step(current_state, noise=apply_noise, feedback=apply_feedback)
        
        # Predict the next state
        prediction = rn.y  # Single output neuron
        
        predictions.append(prediction)
        targets.append(target_state)
        
        # Train the reservoir using the selected method
        if use_lqr:
            current_rec_error = rn.train_recurrent_lqr(target_state, 1, 1)
        else:
            current_rec_error = rn.train_recurrent(target_state)

        current_error = rn.train_readout(target_state)
        total_error += current_error
        
        # Print out error periodically
        if print_step_error and step % 10 == 0:
            print(f"Step {step}: Current error = {current_error:.3f}")
            print(f"Step {step}: Current rec error = {current_rec_error:.3f}")

    # Print average error over the steps
    print(f"Average error: {total_error / max_steps:.3f}")

    return predictions, targets, total_error

# Read Bitcoin price data from the CSV file
df = pd.read_csv('./bitcoin_price_data_hour.csv')
df = pd.read_csv('C:/Users/g4m3r/source/repos/ReservoirComputing/bitcoin_price_data_day.csv')

# Use 'close' prices as the time series
prices = df['close'].values

# Normalize the prices (optional but recommended)
normalized_prices = (prices - np.min(prices)) / (np.max(prices) - np.min(prices))

# Some Signal
t = np.linspace(0, 10, 1000)  # 1000 time steps
signal = np.sin(t)

data = normalized_prices
#data = signal

# Define your network configuration
base_config = {
    'Ni': 1,  # Number of input neurons. Determines how many different input signals the network can handle.
    'N': 100,  # Total number of neurons in the main recurrent layer (reservoir).
    'N_u': 100,  # Total number of neurons in the control input layer. Used for integrating external control signals.
    'No': 1,  # Total number of output neurons. Defines the dimensionality of the network's output.
    'No_u': 1,  # Total number of output neurons in the control output layer.
    'tau': 10,  # Time constant for the neurons in the main reservoir. Influences the decay rate of neuronal activation.
    'tau_u': 10,  # Time constant for the neurons in the control input layer. Affects how quickly these neurons react to changes.
    'g': 1.5,  # Scaling factor for the synaptic weights in the main reservoir. Influences the overall dynamics and stability of the reservoir.
    'g_u': 1.5,  # Scaling factor for the synaptic weights in the control input layer. Controls the impact of control inputs on the network.
    'pc': 1.0,  # Connectivity probability within the main reservoir. Determines the sparsity of the reservoir connections.
    'pc_u': 0.1,  # Connectivity probability within the control input layer. Affects the sparsity of connections in the control network.
    'Io': 1.0,  # Intensity of the noise added to the input. Can be used to simulate real-world variability and improve robustness.
    'P_plastic': 0.2,  # Proportion of plastic (modifiable) synapses in the main reservoir. A value of 1.0 means all synapses are plastic.
    'P_plastic_u': 0.2  # Proportion of plastic synapses in the control input layer. Determines how adaptable the control inputs are.
}

# Initialize the network
rn = RecurrentNetworkLQR2(base_config)
rn2 = RecurrentNetworkLQR2(base_config)

# Train with LQR
print('Train with LQR')
predictions_lqr, targets_lqr, total_error_lqr = train_reservoir_model(rn, data, apply_noise=True, apply_feedback=True, use_lqr=True)

# Train without LQR
print('Train with FORCE')
predictions_no_lqr, targets_no_lqr, total_error_no_lqr = train_reservoir_model(rn2, data, apply_noise=True, apply_feedback=True, use_lqr=False)

# Convert predictions and targets to one-dimensional arrays
predictions_flat_lqr = np.ravel(predictions_lqr)
targets_flat_lqr = np.ravel(targets_lqr)

predictions_flat_no_lqr = np.ravel(predictions_no_lqr)
targets_flat_no_lqr = np.ravel(targets_no_lqr)

print(f"Error with LQR   = {(total_error_lqr/len(data)):.3f}")
print(f"Error with FORCE = {(total_error_no_lqr/len(data)):.3f}")

# Plot predictions and flattened targets for both cases
plt.plot(predictions_flat_lqr, label='Predictions (with LQR)')
plt.plot(predictions_flat_no_lqr, label='Predictions (with FORCE)')
plt.plot(targets_flat_no_lqr, label='Targets')

plt.xlabel('Time Step')
plt.ylabel('Normalized Price')
plt.title('Bitcoin Price Prediction vs. Target')
plt.legend()
plt.show()