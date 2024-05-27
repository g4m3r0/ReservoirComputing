# Reservoir Computing for Chaotic Time Series Prediction

This repository contains the code and results of a series of experiments investigating the performance of reservoir computing models for predicting chaotic time series. The experiments focus on evaluating the impact of various factors, including the presence of noise, the inclusion of thalamus-like feedback, the number of neurons, and a comparison between LQR and FORCE learning methods.

## Experiments

### 1. Impact of Noise Variance on LQR vs FORCE Performance
This experiment evaluates the robustness of LQR and FORCE training methods when exposed to different levels of Gaussian noise (standard deviations: 0.001, 0.01, 0.1, 1.0) during the prediction of the Lorenz attractor time series. The mean squared error (MSE) is used to compute the error between the prediction and the target.

### 2. Comparison of LQR vs FORCE Performance
This experiment compares the performance of LQR and FORCE models under different conditions with varying numbers of neurons (50, 100, 250, 500). The effect of the number of neurons on model performance is analyzed.

### 3. Comparison of Thalamic, None, and Random Feedback
This experiment compares the performance of LQR and FORCE models under three different feedback conditions: thalamic feedback, random thalamic feedback (R2 reservoir weights randomly initialized and untrained), and no feedback.

## Repository Structure
- `Experiment1-py`: Scripts for the impact of noise variance experiment.
- `Experiment2.py`: Scripts and data for the comparison of LQR vs FORCE performance experiment. 
- `Experiment3.py`: Scripts and data for the comparison of thalamic, none, and random feedback experiment. 

- `generate_data.py`: Script for generating Lorenz attractor data.
- `lorenz_attractor.py`: Module containing functions for generating the Lorenz attractor.
- `lorenz_data.npy`: Attractor data used in the experiments.
- `plotLorenzData.py`: Script for plotting the attractor data.
- `reservoir_model.py`: Base script containing the classes for the reservoir computing model.
- `train_reservoir.py`: Script used for training the reservoir computing model. 

- `results/`: Consolidated results and analysis of all experiments. 

- `README.md`: Overview of the repository and experiments.

## Dependencies
- Python 3.10.6 (or later)
- NumPy 1.20.2 (or later)
- MatPlotLib
- Pandas
- Threading
- CSV

## Usage
1. Clone the repository: `git clone https://github.com/g4m3r0/ReservoirComputing.git`
2. Install the required dependencies: `pip install -r requirements.txt`
3. Run the desired experiment scripts: `python Experiment1.py`