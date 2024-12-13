import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.signal import butter, filtfilt, hilbert, savgol_filter

# ----------------------------------------
# Function Definitions
# ----------------------------------------

def compute_and_plot_fft(signal, frame_rate, title="Frequency Spectrum in BPM"):
    """
    Computes and plots the FFT of a signal, displaying frequencies in BPM.
    """
    fft_result = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(fft_result), d=1/frame_rate)
    freqs_bpm = freqs * 60
    fft_magnitude = np.abs(fft_result[:len(fft_result)//2])
    freqs_bpm = freqs_bpm[:len(freqs)//2]
    valid_indices = (freqs_bpm >= 3) & (freqs_bpm < 150)
    fft_magnitude = fft_magnitude[valid_indices]
    freqs_bpm = freqs_bpm[valid_indices]
    plt.rcParams['text.color'] = 'white'
    plt.rcParams['axes.labelcolor'] = 'white'
    plt.rcParams['xtick.color'] = 'white'
    plt.rcParams['ytick.color'] = 'white'
    plt.rcParams['axes.edgecolor'] = 'white'
    plt.rcParams['savefig.facecolor'] = 'none'  
    plt.rcParams['figure.facecolor'] = 'none'   
    plt.rcParams['axes.facecolor'] = 'none'     
    plt.rcParams['legend.edgecolor'] = 'white'
    plt.figure(figsize=(10, 6))
    plt.plot(freqs_bpm, fft_magnitude, label="FFT Magnitude")
    plt.xlabel("Frequency (BPM)")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()
    dominant_freq_bpm = freqs_bpm[np.argmax(fft_magnitude)]
    print(f"Dominant Frequency: {dominant_freq_bpm:.2f} BPM")

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass_filter(signal, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return filtfilt(b, a, signal)

def lowpass_filter(signal, cutoff_freq, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, signal)

# ----------------------------------------
# User-Defined Parameters
# ----------------------------------------

filename = 'Respiration_Justac_3m_30FPS'
delimiter = ','                         
frame_rate = 30.0                       
duration = 60 + 1/3                     
sampling_rate = 23.328e9                
speed_of_light = 3e8                    

# Bandwidth and Range Parameters
bandwidth = 2.5e9                       
Rmax = speed_of_light / (2 * sampling_rate)
range_resolution = Rmax

lowcut_resp = 0.1                      
highcut_resp = 0.5                     
lowcut_heart = 1.0                     
highcut_heart = 2.0                    

# ----------------------------------------
# Load and Preprocess Data
# ----------------------------------------

try:
    raw_data = np.loadtxt(filename, delimiter=delimiter)
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

data = raw_data[:, :-1] if raw_data.shape[1] > 1 else raw_data
num_frames, num_bins = data.shape
distance_axis = np.arange(num_bins) * range_resolution

# Clutter Removal and Normalization
ignore_bins = int(0.5 / range_resolution)
distance_axis = distance_axis[ignore_bins:]
data = data[:, ignore_bins:]
data_no_clutter = data - np.mean(data, axis=0)
data_no_clutter_norm = (data_no_clutter - data_no_clutter.min()) / (data_no_clutter.max() - data_no_clutter.min() + 1e-15)

# ----------------------------------------
# Analyze Variance Across Bins
# ----------------------------------------

variances_all_bins = np.var(data_no_clutter_norm, axis=0)

plt.rcParams['text.color'] = 'white'
plt.rcParams['axes.labelcolor'] = 'white'
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'
plt.rcParams['axes.edgecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'none'  
plt.rcParams['figure.facecolor'] = 'none'   
plt.rcParams['axes.facecolor'] = 'none'     
plt.rcParams['legend.edgecolor'] = 'white'
plt.figure(figsize=(10, 6))
plt.plot(distance_axis, variances_all_bins, label="Variance per Bin")
plt.xlabel("Distance (m)")
plt.ylabel("Variance")
plt.title("Variance Across Bins")
plt.grid(True)
plt.legend()
plt.show()

# Automatically Detect Range of Interest
thresh_variance = 0.01  
selected_bins = np.where(variances_all_bins > thresh_variance)[0]
close_range = (selected_bins[0] + ignore_bins) * range_resolution if len(selected_bins) > 0 else 0.5
far_range = (selected_bins[-1] + ignore_bins) * range_resolution if len(selected_bins) > 0 else num_bins * range_resolution

# Target Bin Selection
start_bin = max(0, int(np.floor(close_range / range_resolution)) - ignore_bins)
end_bin = min(num_bins - 1, int(np.ceil(far_range / range_resolution)) - ignore_bins)
target_bin = start_bin + np.argmax(variances_all_bins[start_bin:end_bin])
target_distance = distance_axis[target_bin]
print(f"Selected target_bin={target_bin}, distance ~ {target_distance:.2f} m")

# ----------------------------------------
# Extract and Analyze Slow-Time Signal
# ----------------------------------------

slow_time_signal = data_no_clutter_norm[:, target_bin]
time_axis = np.arange(num_frames) / frame_rate
slow_time_signal_norm = (slow_time_signal - np.mean(slow_time_signal)) / np.std(slow_time_signal)
compute_and_plot_fft(slow_time_signal_norm, frame_rate, title="Frequency Spectrum of Normalized Signal")

# Envelope Extraction
envelope_signal = np.abs(hilbert(slow_time_signal_norm))
filtered_envelope_signal = lowpass_filter(envelope_signal, cutoff_freq=0.2, fs=frame_rate)

# Plot Envelope
plt.figure(figsize=(10, 6))
plt.plot(time_axis, filtered_envelope_signal, label="Filtered Envelope Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Filtered Envelope Signal (Respiration)")
plt.grid(True)
plt.legend()
plt.show()




# ######################################
# ### Apply Envelope Extraction and Filtering for Specific Bin Range
# ######################################
# def process_and_plot_bins(bin_start, bin_end, cutoff_frequency=0.5):
#     """
#     Processes and plots slow-time signals, envelopes, and filtered envelopes
#     for a specified range of bins.
    
#     Args:
#     - bin_start (int): Starting bin number.
#     - bin_end (int): Ending bin number.
#     - cutoff_frequency (float): Low-pass filter cutoff frequency (Hz).
#     """
#     # Ensure valid range
#     if bin_start < 0 or bin_end >= num_bins or bin_start > bin_end:
#         print("Invalid bin range. Please enter a valid range.")
#         return

#     # Loop through the bins in the range
#     for bin_idx in range(bin_start, bin_end + 1):
#         # Extract slow-time signal for the current bin
#         slow_time_signal = data_no_clutter_norm[:, bin_idx]

#         # Normalize the slow-time signal
#         slow_time_signal_norm = (slow_time_signal - np.mean(slow_time_signal)) / np.std(slow_time_signal)

#         # Apply Hilbert Transform to extract the envelope
#         envelope_signal = np.abs(hilbert(slow_time_signal_norm))

#         # Apply low-pass filter to the envelope
#         filtered_envelope_signal = lowpass_filter(envelope_signal, cutoff_frequency, frame_rate)

#         compute_and_plot_fft(
#             envelope_signal,
#             frame_rate,
#             title="Frequency Spectrum of Filtered Respiration Signal"
#         )

#         # Plot the results for the current bin
#         plt.figure(figsize=(12, 6))
#         plt.plot(time_axis, slow_time_signal_norm, label="Normalized Signal", alpha=0.6)
#         plt.plot(time_axis, envelope_signal, label="Envelope Signal", color='orange', alpha=0.8)
#         plt.plot(time_axis, filtered_envelope_signal, label="Filtered Envelope Signal", color='green', linewidth=1.5)
#         plt.xlabel("Time (s)")
#         plt.ylabel("Amplitude")
#         plt.title(f"Bin {bin_idx}: Signal, Envelope, and Filtered Envelope")
#         plt.legend()
#         plt.grid(True)
#         plt.show()

# # Parameters: Set bin range and cutoff frequency
# bin_start = 186  # Starting bin
# bin_end = 190    # Ending bin
# cutoff_frequency = 0.2  # Low-pass filter cutoff frequency in Hz

# # Call the function to process and plot
# process_and_plot_bins(bin_start, bin_end, cutoff_frequency)







# #######################################
# # Calculate Heart Rate in 1-2 Hz Range
# #######################################

# # Perform FFT on the normalized signal
# fft_result = np.fft.fft(slow_time_signal_norm)
# freqs = np.fft.fftfreq(len(fft_result), d=1/frame_rate)  # Frequency axis
# fft_magnitude = np.abs(fft_result[:len(fft_result)//2])  # Positive half of FFT
# freqs = freqs[:len(freqs)//2]  # Positive frequencies only

# # Isolate frequencies between 1 Hz and 2 Hz
# heart_rate_band = (freqs >= 1.2) & (freqs <= 2.0)  # 1 Hz to 2 Hz range
# heart_rate_freqs = freqs[heart_rate_band]
# heart_rate_amplitudes = fft_magnitude[heart_rate_band]

# # Find the dominant frequency in this band
# if len(heart_rate_freqs) > 0:
#     dominant_freq = heart_rate_freqs[np.argmax(heart_rate_amplitudes)]
#     heart_rate_bpm = dominant_freq * 60  # Convert Hz to BPM
#     print(f"Estimated Heart Rate: {heart_rate_bpm:.2f} BPM")
# else:
#     print("No dominant frequency found in the 1-2 Hz band.")