from lorenz_attractor import LorenzAttractorDataGenerator

# Load the saved data for experiments
data_generator = LorenzAttractorDataGenerator()
lorenz_data = data_generator.load_data('lorenz_data1.npy')

# Print the loaded data
data_generator.print_data(lorenz_data)