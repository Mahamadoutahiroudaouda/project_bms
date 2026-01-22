
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time

# Import our modules
from data_loader import DataLoader
from ekf_estimator import EKFEstimator

def run_ekf_simulation(loader, dt=1.0):
    """
    Runs EKF simulation on the full dataset.
    """
    print("--- Starting EKF Simulation ---")
    data = loader.load_data()
    
    # Get arrays
    currents = data['Current'].values
    voltages = data['Voltage'].values
    true_socs = data['SOC'].values
    
    # Initialize EKF
    # Capacity: Typical 18650 cell ~ 2.5Ah to 3.0Ah. Let's guess 2.5Ah based on init in EKF file
    ekf = EKFEstimator(dt=dt, capacity_Ah=2.5) 
    
    # Initialize x based on first true SOC to be fair/stable, or let it converge
    initial_soc = true_socs[0]
    ekf.x[0,0] = initial_soc
    
    estimated_socs = []
    
    start_time = time.time()
    
    for i in range(len(data)):
        # Measurements
        I = currents[i]
        V = voltages[i]
        
        # EKF Step
        ekf.predict(current=I)
        soc_est = ekf.update(voltage=V, current=I)
        
        estimated_socs.append(soc_est)
        
        if i % 10000 == 0:
            print(f"Index {i}/{len(data)} - True: {true_socs[i]:.4f}, Est: {soc_est:.4f}")
            
    print(f"EKF Simulation finished in {time.time() - start_time:.2f} seconds.")
    
    return true_socs, np.array(estimated_socs)

def evaluate_metrics(true, predicted, method_name="Method"):
    """Calculates and prints metrics."""
    mae = mean_absolute_error(true, predicted)
    rmse = np.sqrt(mean_squared_error(true, predicted))
    r2 = r2_score(true, predicted)
    
    print(f"--- {method_name} Metrics ---")
    print(f"MAE:  {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"R2:   {r2:.6f}")
    
    return mae, rmse, r2

def plot_results(true, predicted, title="SOC Estimation"):
    plt.figure(figsize=(12, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(true, label='True SOC', color='black', linewidth=1.5)
    plt.plot(predicted, label='Estimated SOC', color='red', linestyle='--')
    plt.title(f'{title} - Comparison')
    plt.ylabel('SOC')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    error = true - predicted
    plt.plot(error, label='Error (True - Est)', color='blue')
    plt.title('Estimation Error')
    plt.ylabel('Error')
    plt.xlabel('Time Steps')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}.png")
    print(f"Plot saved as {title.replace(' ', '_')}.png")
    # plt.show() # Blocking, improved to output file in Agent mode

if __name__ == "__main__":
    filepath = r"c:\Users\PC\Desktop\projet bms yahya\project_bms\battery_data_csv_forEstimation.csv"
    loader = DataLoader(filepath)
    
    # 1. Run EKF
    true_soc, ekf_soc = run_ekf_simulation(loader)
    evaluate_metrics(true_soc, ekf_soc, "EKF")
    plot_results(true_soc, ekf_soc, "EKF SOC Estimation")
    
    # Logic for AI models will go here later
