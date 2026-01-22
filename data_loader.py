import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

class DataLoader:
    def __init__(self, filepath, seq_length=50):
        self.filepath = filepath
        self.seq_length = seq_length
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.data = None
        self.feature_columns = ['Current', 'Voltage', 'Temperature']
        self.target_column = ['SOC']

    def load_data(self):
        """Loads data from CSV."""
        print(f"Loading data from {self.filepath}...")
        self.data = pd.read_csv(self.filepath)
        # Drop any NaN values just in case
        self.data = self.data.dropna().reset_index(drop=True)
        print(f"Data loaded. Shape: {self.data.shape}")
        return self.data

    def normalize_data(self):
        """Normalizes features and target."""
        if self.data is None:
            self.load_data()
        
        # Scale features and target
        all_cols = self.feature_columns + self.target_column
        self.data_scaled = self.scaler.fit_transform(self.data[all_cols])
        
        # Convert back to dataframe for easier handling if needed, 
        # but numpy array is usually better for sequence generation
        self.data_scaled = pd.DataFrame(self.data_scaled, columns=all_cols)
        print("Data normalized.")
        return self.data_scaled

    def fit_scaler(self, data):
        """Fits the scaler to the provided data (e.g. training set only to avoid leakage)."""
        all_cols = self.feature_columns + self.target_column
        self.scaler.fit(data[all_cols])

    def transform_data(self, data):
        """Applies normalization."""
        all_cols = self.feature_columns + self.target_column
        return self.scaler.transform(data[all_cols])
        
    def inverse_transform_soc(self, scaled_soc):
        """Converts scaled SOC back to original scale."""
        # Create a dummy array with the same shape as original scaler input
        # We only care about the last column (SOC)
        dummy_data = np.zeros((len(scaled_soc), len(self.feature_columns) + 1))
        dummy_data[:, -1] = scaled_soc.flatten()
        
        inverse_data = self.scaler.inverse_transform(dummy_data)
        return inverse_data[:, -1]

    def create_sequences(self, data_array, seq_length):
        """Creates sequences for LSTM: [samples, time_steps, features]."""
        xs, ys = [], []
        # data_array structure: [Current, Voltage, Temperature, SOC]
        # features are cols 0, 1, 2
        # target is col 3 (SOC)
        
        for i in range(len(data_array) - seq_length):
            x = data_array[i:(i + seq_length), :-1] # Features
            y = data_array[i + seq_length, -1]      # Target (Next SOC step)
            xs.append(x)
            ys.append(y)
            
        return np.array(xs), np.array(ys)

    def get_train_test_data(self, test_size=0.2):
        """
        Splits data into train and test sets *before* shuffling or creating sequences 
        to preserve time continuity for testing on unseen future data (optional), 
        or random split. For battery data, usually we want to test on a separate drive cycle 
        or a later part of the data. 
        However, specs say "Profil de courant composé de deux séquences".
        We should probably split by time to be realistic (train on first part, test on second).
        """
        if self.data is None:
            self.load_data()

        # Simple time-based split
        split_index = int(len(self.data) * (1 - test_size))
        
        train_df = self.data.iloc[:split_index]
        test_df = self.data.iloc[split_index:]
        
        # Fit scaler on TRAIN data only
        self.fit_scaler(train_df)
        
        train_scaled = self.transform_data(train_df)
        test_scaled = self.transform_data(test_df)
        
        # Create sequences
        X_train, y_train = self.create_sequences(train_scaled, self.seq_length)
        X_test, y_test = self.create_sequences(test_scaled, self.seq_length)
        
        return X_train, y_train, X_test, y_test

if __name__ == "__main__":
    # Test the loader
    filepath = r"c:\Users\PC\Desktop\projet bms yahya\project_bms\battery_data_csv_forEstimation.csv"
    loader = DataLoader(filepath, seq_length=50)
    data = loader.load_data()
    print(data.head())
    
    X_train, y_train, X_test, y_test = loader.get_train_test_data()
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")
