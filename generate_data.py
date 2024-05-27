from lorenz_attractor import LorenzAttractorDataGenerator

# Generate Lorenz attractor data for a single dimension
dimension = 0
data_generator = LorenzAttractorDataGenerator()
lorenz_data = data_generator.generate_data(dimension)

# Save the generated data to a file
data_filename = './lorenz_data1.npy'
data_generator.save_data(lorenz_data, data_filename)

# Print data
data_generator.print_data(lorenz_data)