
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time

from data_loader import DataLoader
from ekf_estimator import EKFEstimator
from lstm_model import LSTMEstimator

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


def run_lstm_estimation(loader, epochs=50, batch_size=64):
    """
    Train and run LSTM-based SOC estimation.
    """
    print("\n--- Starting LSTM Training ---")
    
    # Get train/test data with sequences
    X_train, y_train, X_test, y_test = loader.get_train_test_data(test_size=0.2)
    
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    print(f"Sequence length: {X_train.shape[1]}")
    print(f"Features: {X_train.shape[2]}")
    
    # Split training for validation
    val_split = int(0.9 * len(X_train))
    X_val, y_val = X_train[val_split:], y_train[val_split:]
    X_train, y_train = X_train[:val_split], y_train[:val_split]
    
    # Create and train LSTM
    lstm = LSTMEstimator(
        seq_length=loader.seq_length, 
        n_features=len(loader.feature_columns),
        lstm_units=64,
        dropout_rate=0.2
    )
    
    start_time = time.time()
    lstm.train(
        X_train, y_train,
        X_val=X_val, y_val=y_val,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    print(f"LSTM Training finished in {time.time() - start_time:.2f} seconds.")
    
    # Predict on test set
    y_pred_scaled = lstm.predict(X_test)
    
    # Inverse transform to original scale
    y_test_original = loader.inverse_transform_soc(y_test)
    y_pred_original = loader.inverse_transform_soc(y_pred_scaled)
    
    return y_test_original, y_pred_original, lstm


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


def plot_comparison(ekf_true, ekf_pred, lstm_true, lstm_pred, ekf_metrics, lstm_metrics):
    """Create side-by-side comparison plot of EKF vs LSTM."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # EKF Results
    axes[0, 0].plot(ekf_true, label='True SOC', color='black', linewidth=1)
    axes[0, 0].plot(ekf_pred, label='EKF Estimated', color='red', linestyle='--', alpha=0.8)
    axes[0, 0].set_title(f'EKF Estimation (MAE: {ekf_metrics[0]:.4f}, R²: {ekf_metrics[2]:.4f})')
    axes[0, 0].set_ylabel('SOC')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # LSTM Results
    axes[0, 1].plot(lstm_true, label='True SOC', color='black', linewidth=1)
    axes[0, 1].plot(lstm_pred, label='LSTM Estimated', color='green', linestyle='--', alpha=0.8)
    axes[0, 1].set_title(f'LSTM Estimation (MAE: {lstm_metrics[0]:.4f}, R²: {lstm_metrics[2]:.4f})')
    axes[0, 1].set_ylabel('SOC')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # EKF Error
    ekf_error = ekf_true - ekf_pred
    axes[1, 0].plot(ekf_error, color='red', alpha=0.7)
    axes[1, 0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[1, 0].set_title(f'EKF Error (RMSE: {ekf_metrics[1]:.4f})')
    axes[1, 0].set_ylabel('Error')
    axes[1, 0].set_xlabel('Time Steps')
    axes[1, 0].grid(True)
    
    # LSTM Error
    lstm_error = lstm_true - lstm_pred
    axes[1, 1].plot(lstm_error, color='green', alpha=0.7)
    axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[1, 1].set_title(f'LSTM Error (RMSE: {lstm_metrics[1]:.4f})')
    axes[1, 1].set_ylabel('Error')
    axes[1, 1].set_xlabel('Time Steps')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig("EKF_vs_LSTM_Comparison.png", dpi=150)
    print("Comparison plot saved as EKF_vs_LSTM_Comparison.png")


if __name__ == "__main__":
    filepath = r"c:\Users\PC\Desktop\projet bms yahya\project_bms\battery_data_csv_forEstimation.csv"
    loader = DataLoader(filepath, seq_length=50)
    
    # 1. Run EKF
    print("="*60)
    print("PART 1: Extended Kalman Filter (EKF)")
    print("="*60)
    ekf_true, ekf_pred = run_ekf_simulation(loader)
    ekf_metrics = evaluate_metrics(ekf_true, ekf_pred, "EKF")
    plot_results(ekf_true, ekf_pred, "EKF_SOC_Estimation")
    
    # 2. Run LSTM
    print("\n" + "="*60)
    print("PART 2: LSTM Neural Network")
    print("="*60)
    lstm_true, lstm_pred, lstm_model = run_lstm_estimation(loader, epochs=50)
    lstm_metrics = evaluate_metrics(lstm_true, lstm_pred, "LSTM")
    plot_results(lstm_true, lstm_pred, "LSTM_SOC_Estimation")
    
    # 3. Comparison
    print("\n" + "="*60)
    print("PART 3: Comparison Summary")
    print("="*60)
    print(f"\n{'Method':<10} {'MAE':<12} {'RMSE':<12} {'R²':<12}")
    print("-"*46)
    print(f"{'EKF':<10} {ekf_metrics[0]:<12.6f} {ekf_metrics[1]:<12.6f} {ekf_metrics[2]:<12.6f}")
    print(f"{'LSTM':<10} {lstm_metrics[0]:<12.6f} {lstm_metrics[1]:<12.6f} {lstm_metrics[2]:<12.6f}")
    
    # Create comparison plot (use test portion of EKF for fair comparison)
    # Note: EKF runs on full data, LSTM only on test set - adjust indices
    test_start_idx = int(len(ekf_true) * 0.8) + loader.seq_length
    ekf_test_true = ekf_true[test_start_idx:test_start_idx + len(lstm_true)]
    ekf_test_pred = ekf_pred[test_start_idx:test_start_idx + len(lstm_pred)]
    
    plot_comparison(ekf_test_true, ekf_test_pred, lstm_true, lstm_pred, ekf_metrics, lstm_metrics)
    
    print("\n✅ All estimations complete!")
