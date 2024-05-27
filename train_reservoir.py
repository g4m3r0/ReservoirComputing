import numpy as np

def train_reservoir_model(rn, data, apply_noise=True, apply_feedback=False, random_feedback=False, use_lqr=False, print_step_error=False, lqr_q_val=1, lqr_r_val=1):
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
    errors = []

    # Train the network to follow the input time series
    max_steps = len(data) - 1
    total_error = 0  # For performance evaluation
    for step in range(max_steps):
        # Get the current target state from the input data
        target_state = data[step+1]
        current_state = data[step]

        # Update network and apply control as usual
        rn.step(current_state, noise=apply_noise, feedback=apply_feedback)
        
        # Predict the next state
        prediction = rn.y[0][0]  # Single output neuron
        
        predictions.append(prediction)
        targets.append(target_state)
        
        # Train the reservoir using the selected method
        current_rec_error = 0
        if use_lqr:
            current_rec_error = rn.train_recurrent_lqr(target_state, lqr_q_val, lqr_r_val)
        else:
            current_rec_error = rn.train_recurrent(target_state)

        # Train thalamus (R2) if random feedback is disabled
        if random_feedback == False:
            if use_lqr:
                current_rec_error_thalamus = rn.train_recurrent_thalamus_lqr(target_state, lqr_q_val, lqr_r_val)
            else:
                current_rec_error_thalamus = rn.train_recurrent_thalamus(target_state)

        current_error = rn.train_readout(target_state)
        total_error += current_error
        errors.append(current_error)
        
        # Print out error periodically
        if print_step_error and step % 10 == 0:
            print(f"Step {step}: Current error = {current_error:.3f}")
            print(f"Step {step}: Current rec error = {current_rec_error:.3f}")

    # Calculate MSE
    mse = np.mean(np.square(np.array(predictions) - np.array(targets)))
    
    # Print average error over the steps
    avg_error = total_error / max_steps
    
    if use_lqr:
        print(f"Average error with LQR: {avg_error:.3f}")
        print(f"MSE with LQR: {mse:.3f}")
    else:
        print(f"Average error with FORCE: {avg_error:.3f}")
        print(f"MSE with FORCE: {mse:.3f}")

    return predictions, targets, errors, total_error, avg_error, mse