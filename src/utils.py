import pandas as pd
import numpy as np
import os
from scipy.signal import butter, lfilter

def butter_lowpass(cutoff, fs, order=4):
    """Design a Butterworth low-pass filter."""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def apply_filter(data, cutoff=10.0, fs=50.0):
    """Apply a Butterworth low-pass filter to the data."""
    b, a = butter_lowpass(cutoff, fs)
    return lfilter(b, a, data, axis=0) # Switch to lfilter to match Android

def create_windows(data, labels, window_size=100, step=50):
    """Create rolling windows from the data."""
    X, y = [], []
    for start in range(0, len(data) - window_size, step):
        end = start + window_size
        X.append(data[start:end])
        # Case where labels might be provided as a sequence
        if isinstance(labels, (list, np.ndarray)) and len(labels) == len(data):
            window_labels = labels[start:end]
            y.append(np.bincount(window_labels).argmax())
        else:
            # Case where labels is a single value for the whole sequence
            y.append(labels)
    return np.array(X), np.array(y)

def load_sensor_data(sample_folder):
    """Load Accelerometer and Gyroscope data from a specific sample folder."""
    accel_path = os.path.join(sample_folder, 'Accelerometer.csv')
    gyro_path = os.path.join(sample_folder, 'Gyroscope.csv')
    
    accel_df = pd.read_csv(accel_path)
    gyro_df = pd.read_csv(gyro_path)
    
    # Merge on seconds_elapsed or assume they are aligned
    # Sensor Logger usually aligns them well if sampled together
    # We'll take the common length if they differ slightly
    min_len = min(len(accel_df), len(gyro_df))
    
    # Selecting x, y, z axes from both
    # Structure: [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]
    data = np.zeros((min_len, 6))
    data[:, 0:3] = accel_df[['x', 'y', 'z']].values[:min_len]
    data[:, 3:6] = gyro_df[['x', 'y', 'z']].values[:min_len]
    
    return data

def load_dataset(base_path):
    """Iterate through the data structure and load all samples."""
    X_all, y_all = [], []
    classes = {'perfect': 1, 'imperfect': 0}
    
    for class_name, class_idx in classes.items():
        class_folder = os.path.join(base_path, class_name)
        if not os.path.exists(class_folder):
            continue
            
        sample_folders = sorted([f for f in os.listdir(class_folder) 
                                if os.path.isdir(os.path.join(class_folder, f))])
        
        for sample_folder in sample_folders:
            path = os.path.join(class_folder, sample_folder)
            data = load_sensor_data(path)
            
            # Preprocessing: Filter
            data_filtered = np.zeros_like(data)
            for i in range(6):
                data_filtered[:, i] = apply_filter(data[:, i])
                
            # Windowing
            X_win, y_win = create_windows(data_filtered, class_idx)
            
            if len(X_win) > 0:
                X_all.append(X_win)
                y_all.append(y_win)
                
    if not X_all:
        return np.array([]), np.array([])
        
    return np.concatenate(X_all), np.concatenate(y_all)
