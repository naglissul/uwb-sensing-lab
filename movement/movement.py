import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, savgol_filter
from scipy.ndimage import maximum_filter1d

#######################################
# Matplotlib Global Settings
#######################################
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
# Load Data
#######################################
data = np.loadtxt("Moving_Justac_5m_30FPS", delimiter=',')
data = data[:, :-1]

#######################################
# RAW 30th frame
#######################################
raw_30th_frame = data[29, :]

# Clutter removal
reference_frame = data[0, :]
mit_filtered_data = reference_frame - data
np.savetxt('mit-filtered-data.txt', mit_filtered_data, delimiter='\t')

#######################################
# Plot Raw and Filtered 30th frame
#######################################
fig = plt.figure()
plt.plot(raw_30th_frame, color='#FFF', label='Raw 30th frame')       
plt.plot(mit_filtered_data[29, :], color='#00FF00', label='MIT filtered 30th frame') 
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.title('Raw and Filtered 30th Frame')
plt.grid(True, color='white', alpha=0.3)
plt.legend(facecolor='none', edgecolor='white', framealpha=0.5)
plt.savefig('Raw_and_filtered_30th_frame.png', dpi=300, transparent=True)
plt.close(fig)

#######################################
# Extract envelope of 30th frame, dropping first 100 samples
#######################################
slow_time_frame_1 = mit_filtered_data[29, 99:]

# Compute Hilbert transform to get analytic signal
analytic_signal = hilbert(slow_time_frame_1)
amplitude_envelope = np.abs(analytic_signal)

# Applying a maximum filter
# envelope = maximum_filter1d(amplitude_envelope, size=5)

# Instead of max filter, we smooth the envelope using Savitzky-Golay
envelope_smoothed = savgol_filter(amplitude_envelope, window_length=11, polyorder=3)

#######################################
# Plot filtered and envelope of 30th frame
#######################################
fig = plt.figure()
plt.plot(slow_time_frame_1, color='#FFF', label='Filtered (30th frame, after drop)')
plt.plot(envelope_smoothed, color='#00FF00', linewidth=2, label='Smoothed Envelope')
plt.xlabel('Sample Index (after drop)')
plt.ylabel('Amplitude')
plt.title('Filtered and Envelope of 30th frame')
plt.grid(True, color='white', alpha=0.3)
plt.legend(facecolor='none', edgecolor='white', framealpha=0.5)
plt.savefig('Filtered_and_envelope_30th_frame.png', dpi=300, transparent=True)
plt.close(fig)

#######################################
# Extract envelope of all frames
#######################################
all_upl = []
for i in range(mit_filtered_data.shape[0]):
    raw_signal_frame = mit_filtered_data[i, :]
    analytic_signal_frame = hilbert(raw_signal_frame)
    amplitude_env_frame = np.abs(analytic_signal_frame)
    # Smooth envelope for each frame as well
    if len(amplitude_env_frame) >= 11:
        frame_envelope = savgol_filter(amplitude_env_frame, window_length=11, polyorder=3)
    else:
        frame_envelope = amplitude_env_frame
    all_upl.append(frame_envelope)

all_upl = np.array(all_upl)
np.savetxt('upper_envelopes.txt', all_upl, delimiter='\t')

#######################################
# Radar parameters 
#######################################
fs = 23.328e9
PRF = 30       
c = 3e8        

X = (np.arange(1, all_upl.shape[1]+1)) * c / (2 * fs)  # Range (m)
Y = (np.arange(1, all_upl.shape[0]+1)) / PRF          # Slow time (s)

#######################################
# Generate 2D Radargram
#######################################
M1 = all_upl

fig = plt.figure()
Y_grid, X_grid = np.meshgrid(Y, X)
plt.pcolormesh(Y_grid, X_grid, M1.T, shading='auto', cmap='jet')
plt.xlabel('Slow time (s)')
plt.ylabel('Range (m)')
plt.title('2D Radargram Plot')
plt.colorbar(label='Amplitude')
plt.savefig('Radargram_of_all_envelopes.png', dpi=300, transparent=True)
plt.close(fig)
