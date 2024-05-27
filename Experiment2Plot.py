import pandas as pd
import matplotlib.pyplot as plt

# Define the data folder path
data_folder_path = "C:\\Users\\g4m3r\\"

# Load the data for each experiment
df_Io001_lqr = pd.read_csv(data_folder_path + 'experiment2_Io.001_lqr_data.csv')
df_Io001_force = pd.read_csv(data_folder_path + 'experiment2_Io.001_force_data.csv')
df_Io01_lqr = pd.read_csv(data_folder_path + 'experiment2_Io.01_lqr_data.csv')
df_Io01_force = pd.read_csv(data_folder_path + 'experiment2_Io.01_force_data.csv')
df_Io_1_lqr = pd.read_csv(data_folder_path + 'experiment2_Io.1_lqr_data.csv')
df_Io_1_force = pd.read_csv(data_folder_path + 'experiment2_Io.1_force_data.csv')
df_Io1_lqr = pd.read_csv(data_folder_path + 'experiment2_Io1_lqr_data.csv')
df_Io1_force = pd.read_csv(data_folder_path + 'experiment2_Io1_force_data.csv')

# Extract the average error for the first row of each iteration
avg_error_Io001_lqr = df_Io001_lqr[df_Io001_lqr['Iteration'].diff() != 0]['Average Error']
avg_error_Io001_force = df_Io001_force[df_Io001_force['Iteration'].diff() != 0]['Average Error']
avg_error_Io01_lqr = df_Io01_lqr[df_Io01_lqr['Iteration'].diff() != 0]['Average Error']
avg_error_Io01_force = df_Io01_force[df_Io01_force['Iteration'].diff() != 0]['Average Error']
avg_error_Io_1_lqr = df_Io_1_lqr[df_Io_1_lqr['Iteration'].diff() != 0]['Average Error']
avg_error_Io_1_force = df_Io_1_force[df_Io_1_force['Iteration'].diff() != 0]['Average Error']
avg_error_Io1_lqr = df_Io1_lqr[df_Io1_lqr['Iteration'].diff() != 0]['Average Error']
avg_error_Io1_force = df_Io1_force[df_Io1_force['Iteration'].diff() != 0]['Average Error']

# Create a new figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Plot the average error for each LQR experiment on the first subplot
ax1.plot(avg_error_Io001_lqr.values, label='Io=0.001 LQR')
ax1.plot(avg_error_Io01_lqr.values, label='Io=0.01 LQR')
ax1.plot(avg_error_Io_1_lqr.values, label='Io=0.1 LQR')
ax1.plot(avg_error_Io1_lqr.values, label='Io=1 LQR')

# Plot the average error for each FORCE experiment on the second subplot
ax2.plot(avg_error_Io001_force.values, label='Io=0.001 FORCE')
ax2.plot(avg_error_Io01_force.values, label='Io=0.01 FORCE')
ax2.plot(avg_error_Io_1_force.values, label='Io=0.1 FORCE')
ax2.plot(avg_error_Io1_force.values, label='Io=1 FORCE')

# Determine the global y-axis limits
y_min = min(min(avg_error_Io001_lqr), min(avg_error_Io01_lqr), min(avg_error_Io_1_lqr), min(avg_error_Io1_lqr),
            min(avg_error_Io001_force), min(avg_error_Io01_force), min(avg_error_Io_1_force), min(avg_error_Io1_force))
y_max = max(max(avg_error_Io001_lqr), max(avg_error_Io01_lqr), max(avg_error_Io_1_lqr), max(avg_error_Io1_lqr),
            max(avg_error_Io001_force), max(avg_error_Io01_force), max(avg_error_Io_1_force), max(avg_error_Io1_force))

# Set the y-axis limits for both subplots
ax1.set_ylim([y_min, y_max])
ax2.set_ylim([y_min, y_max])

# Add a legend to each subplot
ax1.legend()
ax2.legend()

# Add grid lines
ax1.grid(True)
ax2.grid(True)

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
mean_error_Io001_lqr = avg_error_Io001_lqr.mean()
mean_error_Io001_force = avg_error_Io001_force.mean()
mean_error_Io01_lqr = avg_error_Io01_lqr.mean()
mean_error_Io01_force = avg_error_Io01_force.mean()
mean_error_Io_1_lqr = avg_error_Io_1_lqr.mean()
mean_error_Io_1_force = avg_error_Io_1_force.mean()
mean_error_Io1_lqr = avg_error_Io1_lqr.mean()
mean_error_Io1_force = avg_error_Io1_force.mean()

# Print out the average error over all iterations
print("Average Error over all iterations:")
print("LQR Io=0.001: ", mean_error_Io001_lqr)
print("FORCE Io=0.001: ", mean_error_Io001_force)
print("LQR Io=0.01: ", mean_error_Io01_lqr)
print("FORCE Io=0.01: ", mean_error_Io01_force)
print("LQR Io=0.1: ", mean_error_Io_1_lqr)
print("FORCE Io=0.1: ", mean_error_Io_1_force)
print("LQR Io=1: ", mean_error_Io1_lqr)
print("FORCE Io=1: ", mean_error_Io1_force)