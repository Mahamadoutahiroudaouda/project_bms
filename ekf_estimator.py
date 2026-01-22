
import numpy as np

class EKFEstimator:
    def __init__(self, dt=1.0, capacity_Ah=2.5):
        """
        Initializes the EKF Estimator.
        
        Args:
            dt (float): Sampling time in seconds.
            capacity_Ah (float): Battery capacity in Ampere-hours.
        """
        self.dt = dt
        self.capacity = capacity_Ah * 3600  # Convert to Coulombs (As)
        
        # Battery Parameters (Thevenin Model)
        # These should ideally be lookup tables f(SOC, Temp), but we use constants for simplicity
        # or estimations if no OCV curve is provided.
        # Assuming generic Li-ion parameters:
        self.R0 = 0.01  # Ohms (Internal Resistance)
        self.R1 = 0.02  # Ohms (Polarization Resistance)
        self.C1 = 2000  # Farads (Polarization Capacitance)
        
        # State Vector: x = [SOC, Vc1]
        self.x = np.array([[0.5], [0.0]]) # Initial guess: 50% SOC, 0V polarization
        
        # Covariance Matrix P
        self.P = np.diag([1.0, 1.0]) * 1e-3
        
        # Process Noise Covariance Q
        self.Q = np.diag([1e-6, 1e-4]) 
        
        # Measurement Noise Covariance R
        self.R = 0.1
        
    def ocv_func(self, soc):
        """
        Open Circuit Voltage (OCV) vs SOC curve.
        Approximated polynomial for a typical Li-ion NMC cell.
        Ideally this comes from Experimental Data (OCV-SOC test).
        """
        # 6th order polynomial approximation
        # Coefficients are examples, might need tuning based on the specific cell chemistry in the PDF if provided.
        # Since no OCV file was explicitly mentioned as separate, we approximate.
        # Commonly used approximation:
        # V = K0 + K1*s + K2/s + K3*ln(s) + K4*ln(1-s)
        # But simple polynomial is often stable enough for demonstration.
        
        # Let's use a robust relationship or a simple linear + log model:
        # Combined Model: V_ocv(z) = K0 - dV/z - K2*z + K3*ln(z) + K4*ln(1-z)
        # Simplified for project:
        soc = np.clip(soc, 0.01, 0.99) # Avoid log(0)
        
        # Example coefficients for Li-ion
        return 3.0 + 1.0 * soc # VERY BASIC LINEAR APPROX if no data.
        # Let's try to be a bit more realistic for Li-ion (3.0V to 4.2V)
        # 3.2V at 0%, 4.2V at 100% -> roughly linear but with curves at ends.
        
        # Better Polynomial (generic):
        # 3.14 + 1.0*soc (Linear baseline) can work for testing logic.
        # Update: Let's use the 'Combined Model' coeffs often found in literature for broad li-ion
        # But strictly speaking, we should have OCV data.
        # If the project provided parameters are inside slx, we might not have them easily.
        # I will build a generic curve:
        # 0% -> 3.2V, 100% -> 4.2V
        return 3.2 + 1.0 * soc - 0.05 * (1/soc) - 0.05 * np.log(1 - soc) if False else 3.2 + (4.2-3.2)*soc

    def ocv_derivative(self, soc):
        """ Derivative of OCV with respect to SOC (dOCV/dSOC). """
        # For linear approx: 3.2 + 1.0*soc
        return 1.0 # (4.2-3.2) if linear

    def predict(self, current):
        """
        Predict step of EKF.
        x_k|k-1 = f(x_k-1, u_k-1)
        P_k|k-1 = A * P_k-1 * A.T + Q
        
        current: Input current (Amps). Positive = Discharging (load), Negative = Charging.
        NOTE: Check convention. Often Discharge is positive in BMS.
        """
        soc = self.x[0, 0]
        v_c1 = self.x[1, 0]
        
        # State Transition Equations
        # SOC_k = SOC_k-1 - (dt / Capacity) * Current
        # Vc1_k = Vc1_k-1 * exp(-dt / (R1*C1)) + R1 * (1 - exp(-dt / (R1*C1))) * Current
        
        alpha = np.exp(-self.dt / (self.R1 * self.C1))
        
        new_soc = soc - (self.dt / self.capacity) * current
        new_v_c1 = v_c1 * alpha + self.R1 * (1 - alpha) * current
        
        self.x = np.array([[new_soc], [new_v_c1]])
        
        # Jacobian A = df/dx
        # df1/dSOC = 1, df1/dVc1 = 0
        # df2/dSOC = 0, df2/dVc1 = alpha
        
        A = np.array([
            [1.0, 0.0],
            [0.0, alpha]
        ])
        
        self.P = A @ self.P @ A.T + self.Q

    def update(self, voltage, current):
        """
        Update step of EKF.
        Correction based on measurement residual.
        """
        soc_pred = self.x[0, 0]
        v_c1_pred = self.x[1, 0]
        
        # Measurement Equation (Output)
        # V_term = OCV(SOC) - Vc1 - R0 * Current
        # (Assuming Discharge Current is Positive => Voltage Drop subtracts)
        
        y_pred = self.ocv_func(soc_pred) - v_c1_pred - self.R0 * current
        
        # Residual
        y_residual = voltage - y_pred
        
        # Jacobian C = dh/dx
        # dh/dSOC = dOCV/dSOC
        # dh/dVc1 = -1
        
        d_ocv = self.ocv_derivative(soc_pred)
        C = np.array([[d_ocv, -1.0]])
        
        # Kalman Gain
        # K = P * C.T * inv(C * P * C.T + R)
        
        S = C @ self.P @ C.T + self.R
        K = self.P @ C.T @ np.linalg.inv(S)
        
        # State Correct
        self.x = self.x + K * y_residual
        
        # Covariance Correct
        I = np.eye(2)
        self.P = (I - K @ C) @ self.P
        
        return self.x[0, 0] # Return estimated SOC
