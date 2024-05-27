import pandas as pd
import matplotlib.pyplot as plt

# Define the data folder path
#data_folder_path = "/data/"
data_folder_path = "C:\\Users\\g4m3r\\"

# Load the data for each experiment
df_N50_lqr = pd.read_csv(data_folder_path + 'experiment1_N50_lqr_data.csv')
df_N50_force = pd.read_csv(data_folder_path + 'experiment1_N50_force_data.csv')
df_N100_lqr = pd.read_csv(data_folder_path + 'experiment1_N100_lqr_data.csv')
df_N100_force = pd.read_csv(data_folder_path + 'experiment1_N100_force_data.csv')
df_N250_lqr = pd.read_csv(data_folder_path + 'experiment1_N250_lqr_data.csv')
df_N250_force = pd.read_csv(data_folder_path + 'experiment1_N250_force_data.csv')
df_N500_lqr = pd.read_csv(data_folder_path + 'experiment1_N500_lqr_data.csv')
df_N500_force = pd.read_csv(data_folder_path + 'experiment1_N500_force_data.csv')

# Extract the average error for the first row of each iteration
avg_error_N50_lqr = df_N50_lqr[df_N50_lqr['Iteration'].diff() != 0]['Average Error']
avg_error_N50_force = df_N50_force[df_N50_force['Iteration'].diff() != 0]['Average Error']
avg_error_N100_lqr = df_N100_lqr[df_N100_lqr['Iteration'].diff() != 0]['Average Error']
avg_error_N100_force = df_N100_force[df_N100_force['Iteration'].diff() != 0]['Average Error']
avg_error_N250_lqr = df_N250_lqr[df_N250_lqr['Iteration'].diff() != 0]['Average Error']
avg_error_N250_force = df_N250_force[df_N250_force['Iteration'].diff() != 0]['Average Error']
avg_error_N500_lqr = df_N500_lqr[df_N500_lqr['Iteration'].diff() != 0]['Average Error']
avg_error_N500_force = df_N500_force[df_N500_force['Iteration'].diff() != 0]['Average Error']

# Create a new figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Plot the average error for each LQR experiment on the first subplot
ax1.plot(avg_error_N50_lqr.values, label='N=50 LQR')
ax1.plot(avg_error_N100_lqr.values, label='N=100 LQR')
ax1.plot(avg_error_N250_lqr.values, label='N=250 LQR')
ax1.plot(avg_error_N500_lqr.values, label='N=500 LQR')

# Plot the average error for each FORCE experiment on the second subplot
ax2.plot(avg_error_N50_force.values, label='N=50 FORCE')
ax2.plot(avg_error_N100_force.values, label='N=100 FORCE')
ax2.plot(avg_error_N250_force.values, label='N=250 FORCE')
ax2.plot(avg_error_N500_force.values, label='N=500 FORCE')

# Determine the global y-axis limits
y_min = min(avg_error_N50_lqr.min(), avg_error_N100_lqr.min(), avg_error_N250_lqr.min(), avg_error_N500_lqr.min(),
            avg_error_N50_force.min(), avg_error_N100_force.min(), avg_error_N250_force.min(), avg_error_N500_force.min())
y_max = max(avg_error_N50_lqr.max(), avg_error_N100_lqr.max(), avg_error_N250_lqr.max(), avg_error_N500_lqr.max(),
            avg_error_N50_force.max(), avg_error_N100_force.max(), avg_error_N250_force.max(), avg_error_N500_force.max())

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

############################ Second Plot ####################################

# Create a new figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Plot the average error for N=500 experiments on the first subplot
ax1.plot(avg_error_N500_lqr.values, label='N500 LQR')
ax1.plot(avg_error_N500_force.values, label='N500 FORCE')

# Assuming 'Prediction' and 'Target' are columns in your data
# Plot the prediction vs target for N=500 experiments on the second subplot
ax2.plot(df_N500_lqr[df_N500_lqr['Iteration'].diff() != 0]['Prediction'].values, label='N500 LQR Prediction')
ax2.plot(df_N500_lqr[df_N500_lqr['Iteration'].diff() != 0]['Target'].values, label='N500 LQR Target')
ax2.plot(df_N500_force[df_N500_force['Iteration'].diff() != 0]['Prediction'].values, label='N500 FORCE Prediction')
ax2.plot(df_N500_force[df_N500_force['Iteration'].diff() != 0]['Target'].values, label='N500 FORCE Target')

# Add a legend to each subplot
ax1.legend()
ax2.legend()

# Add labels and title to each subplot
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Average Error')
ax1.set_title('Average Error for N=500 Experiments')

ax2.set_xlabel('Iteration')
ax2.set_ylabel('Value')
ax2.set_title('Prediction vs Target for N=500 Experiments')

# Show the plot
plt.show()

# Calculate the average error over all iterations for each experiment
mean_error_N50_lqr = avg_error_N50_lqr.mean()
mean_error_N50_force = avg_error_N50_force.mean()
mean_error_N100_lqr = avg_error_N100_lqr.mean()
mean_error_N100_force = avg_error_N100_force.mean()
mean_error_N250_lqr = avg_error_N250_lqr.mean()
mean_error_N250_force = avg_error_N250_force.mean()
mean_error_N500_lqr = avg_error_N500_lqr.mean()
mean_error_N500_force = avg_error_N500_force.mean()

# Print out the average error over all iterations
print("Average Error over all iterations:")
print("LQR N=50: ", mean_error_N50_lqr)
print("FORCE N=50: ", mean_error_N50_force)
print("LQR N=100: ", mean_error_N100_lqr)
print("FORCE N=100: ", mean_error_N100_force)
print("LQR N=250: ", mean_error_N250_lqr)
print("FORCE N=250: ", mean_error_N250_force)
print("LQR N=500: ", mean_error_N500_lqr)
print("FORCE N=500: ", mean_error_N500_force)