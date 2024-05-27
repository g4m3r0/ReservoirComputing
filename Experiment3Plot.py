import pandas as pd
import matplotlib.pyplot as plt

# Define the data folder path
data_folder_path = "C:\\Users\\g4m3r\\"

# Load the data for each experiment
df_NoFeedback_lqr = pd.read_csv(data_folder_path + 'experiment3_NoFeedback_lqr_data.csv')
df_NoFeedback_force = pd.read_csv(data_folder_path + 'experiment3_NoFeedback_force_data.csv')
df_Random_lqr = pd.read_csv(data_folder_path + 'experiment3_Random_lqr_data.csv')
df_Random_force = pd.read_csv(data_folder_path + 'experiment3_Random_force_data.csv')
df_Thalamic_lqr = pd.read_csv(data_folder_path + 'experiment3_Thalamic_lqr_data.csv')
df_Thalamic_force = pd.read_csv(data_folder_path + 'experiment3_Thalamic_force_data.csv')

# Extract the average error for the first row of each iteration
avg_error_NoFeedback_lqr = df_NoFeedback_lqr[df_NoFeedback_lqr['Iteration'].diff() != 0]['Average Error']
avg_error_NoFeedback_force = df_NoFeedback_force[df_NoFeedback_force['Iteration'].diff() != 0]['Average Error']
avg_error_Random_lqr = df_Random_lqr[df_Random_lqr['Iteration'].diff() != 0]['Average Error']
avg_error_Random_force = df_Random_force[df_Random_force['Iteration'].diff() != 0]['Average Error']
avg_error_Thalamic_lqr = df_Thalamic_lqr[df_Thalamic_lqr['Iteration'].diff() != 0]['Average Error']
avg_error_Thalamic_force = df_Thalamic_force[df_Thalamic_force['Iteration'].diff() != 0]['Average Error']

# Create a new figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Plot the average error for each LQR experiment on the first subplot
ax1.plot(avg_error_NoFeedback_lqr.values, label='No Feedback')
ax1.plot(avg_error_Random_lqr.values, label='Random Feedback')
ax1.plot(avg_error_Thalamic_lqr.values, label='Thalamic Feedback')

# Plot the average error for each FORCE experiment on the second subplot
ax2.plot(avg_error_NoFeedback_force.values, label='No Feedback')
ax2.plot(avg_error_Random_force.values, label='Random Feedback')
ax2.plot(avg_error_Thalamic_force.values, label='Thalamic Feedback')

# Determine the global y-axis limits
y_min = min(min(avg_error_NoFeedback_lqr), min(avg_error_Random_lqr), min(avg_error_Thalamic_lqr),
            min(avg_error_NoFeedback_force), min(avg_error_Random_force), min(avg_error_Thalamic_force))
y_max = max(max(avg_error_NoFeedback_lqr), max(avg_error_Random_lqr), max(avg_error_Thalamic_lqr),
            max(avg_error_NoFeedback_force), max(avg_error_Random_force), max(avg_error_Thalamic_force))

# Set the y-axis limits for both subplots
ax1.set_ylim([y_min, y_max])
ax2.set_ylim([y_min, y_max])

# Add a legend to each subplot
ax1.legend()
ax2.legend()

# Add labels and title to each subplot
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Average Error')
ax1.set_title('LQR Experiments')

ax2.set_xlabel('Iteration')
ax2.set_ylabel('Average Error')
ax2.set_title('FORCE Experiments')

# Show the plot
plt.show()

############################ Print out average over all iterations ####################################

# Calculate the average error over all iterations for each experiment
mean_error_NoFeedback_lqr = avg_error_NoFeedback_lqr.mean()
mean_error_NoFeedback_force = avg_error_NoFeedback_force.mean()
mean_error_Random_lqr = avg_error_Random_lqr.mean()
mean_error_Random_force = avg_error_Random_force.mean()
mean_error_Thalamic_lqr = avg_error_Thalamic_lqr.mean()
mean_error_Thalamic_force = avg_error_Thalamic_force.mean()

# Print out the average error over all iterations
print("Average Error over all iterations:")
print("LQR No Feedback: ", mean_error_NoFeedback_lqr)
print("FORCE No Feedback: ", mean_error_NoFeedback_force)
print("LQR Random: ", mean_error_Random_lqr)
print("FORCE Random: ", mean_error_Random_force)
print("LQR Thalamic: ", mean_error_Thalamic_lqr)
print("FORCE Thalamic: ", mean_error_Thalamic_force)
