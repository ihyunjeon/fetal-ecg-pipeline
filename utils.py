import os
import numpy as np
import mne
import wfdb
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader
from scipy.signal import find_peaks, butter, filtfilt

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    # Apply filter along time axis (axis=1)
    y = filtfilt(b, a, data, axis=1)
    return y

def create_windows(record_path, window_size=2000, step_size=500):
    """
    Reads an EDF file and slices it into windows.
    Returns:
        X (np.array): (Num_Windows, 4, 2000) -> Abdominal Signals (Input)
        y (np.array): (Num_Windows, 1, 2000) -> Fetal Scalp Signal (Target)
    """
    try:
        raw = mne.io.read_raw_edf(record_path, preload=True, verbose=False)
    except FileNotFoundError:
        print(f"Error: File not found at {record_path}")
        return np.array([]), np.array([])

    # Get all data: Shape is (Channels, Time)
    all_signals = raw.get_data() 
    
    # SEPARATE INPUT AND TARGET
    # Channels 0-3 are Abdominal (Input)
    abdominal_signals = all_signals[:4, :] 
    
    # Channel 4 is Direct Fetal Scalp (Ground Truth Target)
    # We use slicing [4:5, :] to keep the dimension as (1, Time)
    scalp_signal = all_signals[4:5, :] 

    abdominal_signals = butter_bandpass_filter(abdominal_signals, 0.5, 50, fs=1000)
    scalp_signal = butter_bandpass_filter(scalp_signal, 0.5, 50, fs=1000)
    abdominal_signals = (abdominal_signals - np.mean(abdominal_signals, axis=1, keepdims=True)) / (np.std(abdominal_signals, axis=1, keepdims=True) + 1e-6)
    scalp_signal = (scalp_signal - np.mean(scalp_signal, axis=1, keepdims=True)) / (np.std(scalp_signal, axis=1, keepdims=True) + 1e-6)
    
    X_list = []
    y_list = []
    
    total_len = abdominal_signals.shape[1]
    
    # Slide the window
    for start in range(0, total_len - window_size, step_size):
        end = start + window_size
        
        # Extract Abdominal Window (Input)
        x_window = abdominal_signals[:, start:end]
        
        # Extract Scalp Window (Target) - NOW A SIGNAL, NOT A LABEL
        y_window = scalp_signal[:, start:end]
        
        X_list.append(x_window)
        y_list.append(y_window)
        
    return np.array(X_list), np.array(y_list)

def load_dataset(file_list):
    """Loops through a list of files and combines them into one big dataset."""
    all_X = []
    all_y = []
    
    for file_path in file_list:
        print(f"Processing {file_path}...")
        X, y = create_windows(file_path)
        if len(X) > 0:
            all_X.append(X)
            all_y.append(y)
            
    if len(all_X) == 0:
        return None, None
        
    final_X = np.concatenate(all_X, axis=0)
    final_y = np.concatenate(all_y, axis=0)
    
    return final_X, final_y

def get_dataloaders(train_files, test_files, batch_size=8):
    """
    Complete pipeline: Loads files -> Converts to Tensor -> Returns DataLoaders
    """
    print("--- Loading Training Data ---")
    X_train, y_train = load_dataset(train_files)
    
    print("\n--- Loading Test Data ---")
    X_test, y_test = load_dataset(test_files)
    
    if X_train is None or X_test is None:
        print("Error: Could not load data.")
        return None, None

    # CRITICAL FIX: Convert targets (y) to Float (for Regression), not Long (for Class)
    tensor_x_train = torch.tensor(X_train).float()
    tensor_y_train = torch.tensor(y_train).float() 
    
    tensor_x_test = torch.tensor(X_test).float()
    tensor_y_test = torch.tensor(y_test).float()
    
    train_loader = DataLoader(TensorDataset(tensor_x_train, tensor_y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(tensor_x_test, tensor_y_test), batch_size=batch_size, shuffle=False)
    
    print(f"\nSuccess! Train Batch: {len(train_loader)} | Test Batch: {len(test_loader)}")
    return train_loader, test_loader

def visualize_signals(record_path, start_sec=0, duration_sec=5):
    """
    Plots the 4 abdominal channels and the 1 scalp channel.
    Automatically adds RED DOTTED LINES for Fetal QRS peaks.
    """
    record_name = os.path.basename(record_path).replace('.edf', '')
    
    # 1. Read Signals
    try:
        raw = mne.io.read_raw_edf(record_path, preload=True, verbose=False)
    except FileNotFoundError:
        print(f"Error: File not found at {record_path}")
        return

    signals = raw.get_data()
    fs = int(raw.info['sfreq']) 
    
    # 2. Get QRS Indices (Try Annotation File -> Fallback to Auto-Detection)
    qrs_indices = []
    
    # Method A: Try loading official .qrs file
    try:
        base_path = record_path.replace('.edf', '')
        annotation = wfdb.rdann(base_path, 'qrs')
        qrs_indices = annotation.sample
        print("✅ Loaded peaks from .qrs annotation file.")
    except Exception:
        # Method B: Automatic Peak Detection on Scalp Channel (Channel 4)
        print("⚠️ Annotation file not found. Auto-detecting peaks on Scalp channel...")
        scalp_signal = signals[4, :]
        # Find peaks that are at least 0.5 distance apart and have some height
        peaks, _ = find_peaks(scalp_signal, height=np.mean(scalp_signal)*1.5, distance=fs*0.3)
        qrs_indices = peaks

    # 3. Calculate Plot Window
    start_sample = int(start_sec * fs)
    end_sample = int((start_sec + duration_sec) * fs)
    if end_sample > signals.shape[1]: end_sample = signals.shape[1]
    
    time_axis = np.linspace(start_sec, start_sec + duration_sec, end_sample - start_sample)
    
    # 4. Plot
    fig, axes = plt.subplots(5, 1, figsize=(12, 12), sharex=True)
    fig.suptitle(f"Record: {record_name} (Time: {start_sec}s - {start_sec+duration_sec}s)", fontsize=16)

    channel_names = ["Abdomen 1", "Abdomen 2", "Abdomen 3", "Abdomen 4", "Direct Scalp (Target)"]

    for i in range(5):
        ax = axes[i]
        
        # Plot Signal
        ax.plot(time_axis, signals[i, start_sample:end_sample], color='black', linewidth=0.8)
        
        # --- THE RED DOTTED LINES ---
        # Filter for peaks inside this specific time window
        beats_in_window = qrs_indices[(qrs_indices >= start_sample) & (qrs_indices < end_sample)]
        
        for beat_idx in beats_in_window:
            # Convert sample index to time (seconds)
            beat_time = beat_idx / fs
            # Draw the red line
            ax.axvline(x=beat_time, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        
        ax.set_title(channel_names[i], loc='left', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        if i == 4:
            ax.set_facecolor('#f0fff0') # Light green for target
            ax.set_ylabel("Amplitude")

    axes[-1].set_xlabel("Time (seconds)")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()