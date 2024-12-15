import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, butter, filtfilt

plt.rcParams['text.color'] = 'white'
plt.rcParams['axes.labelcolor'] = 'white'
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'
plt.rcParams['axes.edgecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'none' 
plt.rcParams['figure.facecolor'] = 'none'   
plt.rcParams['axes.facecolor'] = 'none'    
plt.rcParams['legend.edgecolor'] = 'white'

#######################################
# Parameters
#######################################
filename = 'TP2_5mins.txt'   
frame_rate = 30.0            
duration = 303.0 + 2.0/15            
min_apnea_duration = 7.0     
threshold_factor = 0.2       

#######################################
# Loading and preprocessing data
#######################################
try:
    raw_data = np.genfromtxt(filename, delimiter=',', dtype='str')
    raw_data = np.char.strip(raw_data, ',')
    raw_data[raw_data == ''] = 'nan'
    data = np.nan_to_num(np.char.replace(raw_data, ',', '').astype(float))
except Exception as e:
    print(f"Error loading or processing data: {e}")
    exit()

num_frames, num_bins = data.shape
print(f"Data loaded: {num_frames} frames, {num_bins} bins")

if num_frames != int(frame_rate * duration):
    print("Warning: Frame count mismatch with expected duration")

#######################################
# Clutter Removal
#######################################
mean_frame = np.mean(data, axis=0)
data_no_clutter = data - mean_frame

#######################################
# Variance-based bin selection
#######################################
variances = np.var(data_no_clutter, axis=0)
target_bin = np.argmax(variances)
print(f"Selected target bin based on maximum variance: {target_bin}")

#######################################
# Time Axis Setup
#######################################
time_axis = np.linspace(0, duration, num_frames)

#######################################
# Target signal
#######################################
target_signal = -data_no_clutter[:, target_bin]

#######################################
# Bandpass Filter
#######################################
def bandpass_filter(signal, fs, lowcut, highcut, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered = filtfilt(b, a, signal)
    return filtered

filtered_signal = bandpass_filter(target_signal, frame_rate, 0.1, 0.6)

#######################################
# Envelope Extraction
#######################################
analytic_signal = hilbert(filtered_signal)
envelope = np.abs(analytic_signal)

#######################################
# Thresholding
#######################################
mean_env = np.mean(envelope)
std_env = np.std(envelope) 
threshold = mean_env - threshold_factor * std_env

#######################################
# Detecting Apnea Regions
#######################################
is_below_threshold = envelope < threshold
transitions = np.diff(is_below_threshold.astype(int))
starts = np.where(transitions == 1)[0]
ends = np.where(transitions == -1)[0]

if starts.size > 0 and ends.size > 0:
    if starts[0] > ends[0]:
        starts = np.insert(starts, 0, 0)
    if ends[-1] < starts[-1]:
        ends = np.append(ends, len(envelope) - 1)

min_frames = min_apnea_duration * frame_rate
apnea_intervals = [(s, e) for s, e in zip(starts, ends) if (e - s) >= min_frames]
apnea_regions = np.array(apnea_intervals)

#######################################
# Ploting the Original Data with Detected Apnea
#######################################
plt.figure(figsize=(12, 6))
plt.plot(time_axis, target_signal, color='white', label='Original Signal (No Clutter)')
plt.plot(time_axis, envelope, color='cyan', label='Envelope')

plt.hlines(threshold, 0, duration, color='grey', linestyle='--', label='Threshold')

# Highlight apnea regions
for i, (start, end) in enumerate(apnea_regions):
    plt.axvspan(start / frame_rate, end / frame_rate, color='green', alpha=0.3, 
                label='Detected Apnea' if i == 0 else None)

plt.title('Detection of Sleep Apnea')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.savefig('detection_of_sleep_apnea.png', transparent=True, dpi=300)
plt.show()

#######################################
# Normalized signal figure
#######################################
# signal_mean = np.mean(filtered_signal)
# signal_std = np.std(filtered_signal)
# normalized_signal = (filtered_signal - signal_mean) / signal_std

# plt.figure(figsize=(12, 6))
# plt.plot(time_axis, normalized_signal, label='Normalized Signal', color='blue')
# plt.plot(time_axis, envelope / signal_std, label='Normalized Envelope', color='red')
# plt.hlines((threshold - signal_mean)/signal_std, 0, duration, color='grey', linestyle='--', label='Normalized Threshold')

# for i, (start, end) in enumerate(apnea_regions):
#     plt.axvspan(start / frame_rate, end / frame_rate, color='green', alpha=0.3,
#                 label='Detected Apnea' if i == 0 else None)

# plt.title('Detection of Sleep Apnea (Normalized for Analysis)')
# plt.xlabel('Time (s)')
# plt.ylabel('Normalized Amplitude')
# plt.grid(True)
# plt.legend()
# plt.savefig('detection_of_sleep_apnea_normalized.png', transparent=True, dpi=300)
# plt.show()

#######################################
# Printing Apnea Information
#######################################
if len(apnea_regions) > 0:
    apnea_times = apnea_regions[:, 0] / frame_rate
    apnea_durations = (apnea_regions[:, 1] - apnea_regions[:, 0]) / frame_rate
    print("Detected apneas at times (s):", apnea_times)
    print("Duration of apneas (s):", apnea_durations)
else:
    print("No apneas detected with the current parameters.")

