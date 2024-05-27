import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class LorenzAttractorDataGenerator:
    def __init__(self, sigma=10, rho=38, beta=2.667, initial_state=[1, 1, 1], t_span=[0, 5], points=1000):
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.initial_state = initial_state
        self.t_span = t_span
        self.points = points

    def lorenz(self, t, z):
        x, y, z = z
        dx = self.sigma * (y - x)
        dy = x * (self.rho - z) - y
        dz = x * y - self.beta * z
        return [dx, dy, dz]
    
    def save_data(self, data, filename):
        np.save(filename, data)

    def load_data(self, filename):
        return np.load(filename)
    
    def generate_data(self, dimension=0):
        t_eval = np.linspace(self.t_span[0], self.t_span[1], self.points)
        solution = solve_ivp(self.lorenz, self.t_span, self.initial_state, t_eval=t_eval)
        return solution.y[dimension]

    def print_data(self, data, dimension=0):
        plt.figure(figsize=(10, 4))
        plt.plot(data, color='blue')
        plt.title(f'Lorenz Attractor - Dimension {dimension}')
        plt.xlabel('Time')
        plt.ylabel(f'Dimension {dimension}')
        plt.show()

if __name__ == '__main__':
  # Test code for showing the default lorez attractor
  data_generator = LorenzAttractorDataGenerator()
  lorenz_data = data_generator.generate_data()
  data_generator.print_data(lorenz_data)