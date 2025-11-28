# SIMULATION-OF-FREQUENCY-DIVISION-MULTIPLEXING-FDM-AND-DEMULTIPLEXING-USING-SCILAB
## AIM:

To write a Scilab program to simulate frequency division multiplexing and demultiplexing for six different frequencies, and verify the demultiplexed outputs correspond to the original signals.

## EQUIPMENTS Needed

Computer with Scilab installed

## ALGORITHM

1.Define six different frequencies to generate six sine wave signals.
2.Generate the time vector to represent time samples.
3.Compute six sine signals for each frequency over the time vector.
4.Frequency Division Multiplexing: sum all six sine signals to make one multiplexed signal.
5.Frequency Division Demultiplexing: for each frequency, multiply the multiplexed signal by a sine wave of that frequency (mixing), then apply a lowpass filter to extract the baseband (original) signal.
6.Plot original signals, multiplexed signal, and demultiplexed signals for verification.

## PROCEDURE

Refer Algorithms and write code for the experiment.
• Open SCILAB in System
• Type your code in New Editor
• Save the file
• Execute the code
• If any Error, correct it in code and execute again
• Verify the generated waveform using Tabulation and Model Waveform

## PROGRAM
```
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# -----------------------------
# Parameters
# -----------------------------
fs = 50000           # Sampling frequency
t = np.arange(0, 0.01, 1/fs)

# Four channel sine signals
ch0 = np.sin(2*np.pi*250*t)
ch1 = np.sin(2*np.pi*500*t)
ch2 = np.sin(2*np.pi*1000*t)
ch3 = np.sin(2*np.pi*2000*t)

# -----------------------------
# Time Division Multiplexing
# -----------------------------
num_channels = 4
samples_per_slot = 5
mux = np.zeros(len(t))

# Generate time slots sequentially
for i in range(0, len(t), num_channels * samples_per_slot):
    for j in range(samples_per_slot):
        if i + j < len(t): mux[i + j] = ch0[i + j]
        if i + samples_per_slot + j < len(t): mux[i + samples_per_slot + j] = ch1[i + samples_per_slot + j]
        if i + 2*samples_per_slot + j < len(t): mux[i + 2*samples_per_slot + j] = ch2[i + 2*samples_per_slot + j]
        if i + 3*samples_per_slot + j < len(t): mux[i + 3*samples_per_slot + j] = ch3[i + 3*samples_per_slot + j]

# -----------------------------
# Demultiplexing
# -----------------------------
demux_ch0 = np.zeros_like(t)
demux_ch1 = np.zeros_like(t)
demux_ch2 = np.zeros_like(t)
demux_ch3 = np.zeros_like(t)

for i in range(0, len(t), num_channels * samples_per_slot):
    for j in range(samples_per_slot):
        if i + j < len(t): demux_ch0[i + j] = mux[i + j]
        if i + samples_per_slot + j < len(t): demux_ch1[i + samples_per_slot + j] = mux[i + samples_per_slot + j]
        if i + 2*samples_per_slot + j < len(t): demux_ch2[i + 2*samples_per_slot + j] = mux[i + 2*samples_per_slot + j]
        if i + 3*samples_per_slot + j < len(t): demux_ch3[i + 3*samples_per_slot + j] = mux[i + 3*samples_per_slot + j]

# -----------------------------
# Low-pass filtering (Reconstruction)
# -----------------------------
def lowpass_filter(sig, cutoff, fs, order=6):
    nyq = 0.5 * fs
    b, a = butter(order, cutoff/nyq, btype='low')
    return filtfilt(b, a, sig)

out0 = lowpass_filter(demux_ch0, 500, fs)
out1 = lowpass_filter(demux_ch1, 1000, fs)
out2 = lowpass_filter(demux_ch2, 2000, fs)
out3 = lowpass_filter(demux_ch3, 3000, fs)

# -----------------------------
# Plotting – Properly spaced
# -----------------------------
plt.figure(figsize=(10,14))
plt.subplots_adjust(top=0.92, hspace=0.8)  # increased vertical spacing

plt.suptitle("FIG 1.3 WAVEFORM - TDM RECEIVER (Simulated)", fontsize=14, y=0.98)

plt.subplot(8,1,1)
plt.plot(t, mux, 'k', linewidth=1)
plt.title("RXD - Multiplexed TDM Signal", pad=10)
plt.ylabel("Amplitude")
plt.grid(True)

plt.subplot(8,1,2)
plt.plot(t, demux_ch0, 'r', linewidth=1)
plt.title("CH0 - Sampled Signal", pad=10)
plt.grid(True)

plt.subplot(8,1,3)
plt.plot(t, out0, 'r', linewidth=1.2)
plt.title("OUT0 - Recovered CH0 Signal", pad=10)
plt.grid(True)

plt.subplot(8,1,4)
plt.plot(t, demux_ch1, 'g', linewidth=1)
plt.title("CH1 - Sampled Signal", pad=10)
plt.grid(True)

plt.subplot(8,1,5)
plt.plot(t, out1, 'g', linewidth=1.2)
plt.title("OUT1 - Recovered CH1 Signal", pad=10)
plt.grid(True)

plt.subplot(8,1,6)
plt.plot(t, demux_ch2, 'b', linewidth=1)
plt.title("CH2 - Sampled Signal", pad=10)
plt.grid(True)

plt.subplot(8,1,7)
plt.plot(t, out2, 'b', linewidth=1.2)
plt.title("OUT2 - Recovered CH2 Signal", pad=10)
plt.grid(True)

plt.subplot(8,1,8)
plt.plot(t, out3, 'm', linewidth=1.2)
plt.title("OUT3 - Recovered CH3 Signal", pad=10)
plt.xlabel("Time (s)")
plt.grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.97])  # ensure no overlap with title
plt.show()
```

## TABULATION:
![f533115a-b22f-4807-994b-6ff0064b8611](https://github.com/user-attachments/assets/16ddf10f-c6b6-45e7-abbc-78abda9737d4)

## GRAPH:

## RESULT:
thus the simulation of frequency division multiplexing fdm and demultiplexing is simulated
