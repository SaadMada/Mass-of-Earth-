import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

# Define the fit function for damped oscillations
def damped_oscillation(t, a, b, d, omega, phi):
    """Model a damped sine wave."""
    return a * np.exp(d * t) * np.sin(omega * t + phi) + b 

# File paths for measurements
folder_path_balls = '/Users/mac/Documents/TP Physics/Earth Mass/Data_wB/'  # With balls
folder_path_no_balls = '/Users/mac/Documents/TP Physics/Earth Mass/Data_nB/'  # Without balls
files_balls = sorted([f for f in os.listdir(folder_path_balls) if f.endswith('.csv')])
files_no_balls = sorted([f for f in os.listdir(folder_path_no_balls) if f.endswith('.csv')])


# Storage for calculated Phi_M and T values
phi_M_values_balls = []
T_values_balls = []
phi_M_values_no_balls = []
T_values_no_balls = []

# Function to process files and calculate Phi_M and T, with plots
def process_files(file_list, folder_path, label):
    phi_M_values = []
    T_values = []
    for file_name in file_list:
        file_path = os.path.join(folder_path, file_name)
        try:
            # Load data from the file
            data = np.loadtxt(file_path, delimiter=',')
            time = data[:, 0]
            angle = data[:, 1]
            
            # Initial parameter guesses for curve fitting
            initial_guess = [np.max(angle), np.mean(angle), -0.01, 2 * np.pi / 295, 0]
            bounds = ([-np.inf, -np.inf, -np.inf, 0, -np.inf], [np.inf, np.inf, 0, np.inf, np.inf])

            # Fit the model to the data
            params, cov = curve_fit(damped_oscillation, time[30:], angle[30:], p0=initial_guess, bounds=bounds)
            a, b, d, omega, phi = params  # Extract fitted parameters

            # Calculate period T from omega
            T = 2 * np.pi / omega
            T_values.append(T)

            # Calculate Phi_M as the mean of the maximum and minimum values in the fitted curve
            fitted_curve = damped_oscillation(time, *params)
            phi_M_deg = (np.max(fitted_curve) + np.min(fitted_curve)) / 2  # Phi_M in degrees
            phi_M_rad = np.radians(phi_M_deg)  # Convert Phi_M to radians
            phi_M_values.append(phi_M_rad)
            # Calculate residuals
            residuals = angle - fitted_curve 
        

            # Plot the data and the fit for verification
            plt.figure()
            plt.plot(time[20:], angle[20:], 'gray', alpha=0.5, label=f'{label} Data')
            plt.plot(time, fitted_curve, 'r--', label='Fitted function')
            plt.xlabel('Time (s)')
            plt.ylabel('Angle (deg)')
            plt.legend()
            plt.title(f'Damped Oscillation Fit - {file_name}')
            plt.show()

            # Plot the residuals
            plt.figure(figsize=(8, 6))
            plt.plot(time[40:], residuals[40:], 'b-', label='Residuals')
            plt.axhline(0, color='k', linestyle='--', linewidth=1)  # Add horizontal line at y=0
            plt.xlabel('Data counts')
            plt.ylabel('Angle (rad)')
            plt.title(f'Residual plot of the measurements - {file_name}')
            plt.legend()
            plt.tight_layout()
            plt.show()


        except Exception as e:
            print(f"Calculation failed for {file_name}: {e}")
    
    return phi_M_values, T_values

# Process files for both cases
phi_M_values_balls, T_values_balls = process_files(files_balls, folder_path_balls, "With Balls")
phi_M_values_no_balls, T_values_no_balls = process_files(files_no_balls, folder_path_no_balls, "Without Balls")

# Calculate averages and errors for both cases
phi_M_values_balls_0 = np.array(phi_M_values_balls[0:7:2])
phi_M_values_balls_1 = np.array(phi_M_values_balls[1:8:2])


a = (phi_M_values_balls_1 - phi_M_values_balls_0) / 2 / 2 # the detector mirror thing

b = (phi_M_values_no_balls[1] - phi_M_values_no_balls[0]) / 2 / 2

meanValue_a = sum(a)/4


delta_phi_M = meanValue_a - b 


stda = np.std(a)
error_delta_phi_M = delta_phi_M*stda/np.mean(a)


phi_M_avg_no_balls = np.mean([abs(x) for x in phi_M_values_no_balls]) / 2
phi_M_error_no_balls = np.std([abs(x) for x in phi_M_values_no_balls]) / 2
T_avg_balls = np.mean(T_values_balls)
T_error_balls = np.std(T_values_balls)
T_avg_no_balls = np.mean(T_values_no_balls)
T_error_no_balls = np.std(T_values_no_balls)


# Calculate Delta Phi_M
# delta_phi_M = phi_M_avg_balls - phi_M_avg_no_balls

# Constants for G calculation
Theta = 2.304e-5  # kg·m², moment of inertia
m = 0.02  # kg, small balls mass
M = 10    # kg, large balls mass
r = 0.024  # m, radius of small balls
R = 0.1    # m, radius of large balls
beta_max = 53.18 * np.pi / 180   # convert degrees to radians
h = 0.2 
b_val = 0.5 * (R / r + r / R)
b_prime = 0.5 * (R / r + r / R + (h**2) / (r * R))  

# Calculate E(beta_max)
a_term = (m * M) * (2 * r * R)**(-0.5)
E_beta_max = a_term * np.sin(beta_max) * ((b_val - np.cos(beta_max))**(-1.5) - (b_prime + np.cos(beta_max))**(-1.5))

# Calculate G using Phi_M with and without balls
G_with_delta_phi_M = (4 * np.pi**2 * Theta * delta_phi_M) / (E_beta_max * T_avg_balls**2)


# Calculate G errors
G_error_with_delta_phi_M = G_with_delta_phi_M * np.sqrt((2 * T_error_balls / T_avg_balls)**2 + (error_delta_phi_M / delta_phi_M )**2)


# Constants for Earth mass calculation
g0 = 9.8008  # m/s², gravitational acceleration at Earth's surface
g0_error = 0.0054 # m/s², error in g0
Radius = 6378137  # m, Earth's radius

# Calculate Earth mass using both G values
EarthM_with_delta_phi_M = g0 * Radius**2 / G_with_delta_phi_M


# Calculate Earth mass errors
EarthM_error_with_delta_phi_M = EarthM_with_delta_phi_M * np.sqrt((g0_error / g0)**2 + (G_error_with_delta_phi_M / G_with_delta_phi_M)**2)


# Display results
print("Delta Phi_M (rad):", delta_phi_M)
print("Phi_M with Balls (rad, after division by 2):",phi_M_values_balls)
print("Phi_M with Balls negative:",phi_M_values_balls_0 )
print("Phi_M with Balls positive:",phi_M_values_balls_1 )
print("Phi_M with Balls (rad, after division by 2):",a)
print("mean value of Phi_M with Balls (rad, after division by 2):",meanValue_a)
print("Phi_M_error with balls is : ", stda)
print("Phi_M without Balls (rad, after division by 2):", phi_M_avg_no_balls)
print("Phi_M_error without balls is : ", phi_M_error_no_balls)
print("Average Period T with balls (s):", T_avg_balls)
print("Average error T with Balls (s) :", T_error_balls )
print("Average Period T with no Balls (s):", T_avg_no_balls)
print("Average error T with no Balls (s) :", T_error_no_balls )
print("E(beta_max):", E_beta_max)
print("Delta Phi_M is:", delta_phi_M )
print("The error of Delta Phi_M is:", error_delta_phi_M )
print("\nCalculated G using Delta Phi_M (m³/kg/s²):", G_with_delta_phi_M)
print("G Error with Delta Phi_M (m³/kg/s²):", G_error_with_delta_phi_M)
print("\nCalculated Earth Mass using Delta Phi_M (kg):", EarthM_with_delta_phi_M)
print("Earth Mass Error with Delta Phi_M (kg):", EarthM_error_with_delta_phi_M)

