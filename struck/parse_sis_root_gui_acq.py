#!/usr/bin/env python3

import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable
from scipy import signal

# ============================================================
# Check Command Line Arguments
# ============================================================
if len(sys.argv) < 4:
    print("Usage: python parse_sis_root_gui_acq.py <channel_to_plot> <fs> <file1.txt> [file2.txt ...]")
    sys.exit(1)

channel_to_plot = int(sys.argv[1])
fs = float(sys.argv[2])
acquisition_filenames = sys.argv[3:]

NUM_BITS_ADC = 16
STR_TO_SEARCH_FOR = "Ch"

data_per_file = {}

# ============================================================
# Parse Files
# ============================================================
for fn in acquisition_filenames:
    print(f"Reading file: {fn}")
    data = {}

    try:
        with open(fn, "r", encoding="utf-8") as f:
            while True:
                index = f.tell()
                line = f.readline()

                if not line:
                    break

                if STR_TO_SEARCH_FOR in line:
                    index = (
                        index
                        + line.find(STR_TO_SEARCH_FOR)
                        + len(STR_TO_SEARCH_FOR)
                    )
                    f.seek(index)

                    content = f.readline().split(";")[:-1]

                    if content:
                        raw_samples = np.array(
                            content[1:],
                            dtype=np.float64,
                        )

                        # Convert ADC counts to mV
                        voltage_samples = (
                            raw_samples / (2 ** (NUM_BITS_ADC - 1))
                            - 1
                        ) * 1000.0

                        data[int(content[0])] = voltage_samples

        data_per_file[fn] = data

    except FileNotFoundError:
        print(f"Error: File '{fn}' not found. Skipping.")

# ============================================================
# Prepare Datasets for Plotting
# ============================================================
datasets = []
labels = []

# Uniform styling for all plotted files
uniform_linestyle = "-"
uniform_linewidth = 1.5

custom_labels = ["Background", "Signal + Noise"]

for fn in acquisition_filenames:
    if fn in data_per_file and channel_to_plot in data_per_file[fn]:
        datasets.append(data_per_file[fn][channel_to_plot])

        labels.append(os.path.basename(fn))
    else:
        print(f"Warning: Channel {channel_to_plot} not found in '{fn}'.")

if not datasets:
    print(f"Error: No valid data found for Channel {channel_to_plot} in the provided files.")
    sys.exit(1)

# ============================================================
# Offset and AC RMS values
# ============================================================
table = PrettyTable()
table.field_names = ["Filename", "Offset [mV]", "AC RMS [mV]"]

for samples, label in zip(datasets, labels):
    offset, rms = np.mean(samples), np.std(samples)
    table.add_row([label, offset, rms])

print(table)

# ============================================================
# 1. Noise Time Decourse
# ============================================================
# plt.figure(figsize=(11, 6))
# for samples, label in zip(datasets, labels):
#     t_array = np.arange(len(samples)) / fs
#     plt.plot(t_array, samples, linestyle=uniform_linestyle, linewidth=uniform_linewidth, alpha=0.85, label=label)
# plt.title(f"Noise Time Decourse - Channel {channel_to_plot}", fontweight="bold")
# plt.xlabel("Time [s]", fontweight="bold")
# plt.ylabel("Amplitude [mV]", fontweight="bold")
# plt.grid(True, linestyle="--", alpha=0.6)
# plt.legend(loc='upper left')
# plt.tight_layout()

# ============================================================
# 2. Amplitude Histogram
# ============================================================
# plt.figure(figsize=(11, 6))
# for samples, label in zip(datasets, labels):
#     plt.hist(samples, bins=50, histtype="step", linewidth=uniform_linewidth, alpha=0.85, label=label)
# plt.title(f"Amplitude Histogram - Channel {channel_to_plot}", fontweight="bold")
# plt.xlabel("Amplitude [mV]", fontweight="bold")
# plt.ylabel("Sample Count", fontweight="bold")
# plt.grid(True, linestyle="--", alpha=0.6)
# plt.legend(loc='upper left')
# plt.tight_layout()

# ============================================================
# 3. Noise FFT
# ============================================================
# plt.figure(figsize=(11, 6))
# for samples, label in zip(datasets, labels):
#     D = np.fft.rfft(samples)
#     f_fft = np.fft.rfftfreq(len(samples), 1 / fs)
#     interest = f_fft > 0
#     plt.semilogx(f_fft[interest], 20 * np.log10(np.abs(D[interest])), linestyle=uniform_linestyle, linewidth=uniform_linewidth, alpha=0.85, label=label)
# plt.title(f"Noise FFT - Channel {channel_to_plot}", fontweight="bold")
# plt.xlabel("Frequency [Hz]", fontweight="bold")
# plt.ylabel("Magnitude [dB]", fontweight="bold")
# plt.grid(True, which="both", linestyle="--", alpha=0.6)
# plt.legend(loc='upper left')
# plt.tight_layout()

# ============================================================
# 4. Welch PSD + Cumulative RMS
# ============================================================
fig, ax = plt.subplots(2, 1, sharex=True, figsize=(11, 8))

fig.suptitle(
    f"SIS8300-KU Noise Analysis - Channel {channel_to_plot}",
    fontsize=14,
    fontweight="bold",
)

for samples, label in zip(datasets, labels):
    nperseg_val = max(256, len(samples) // 100)

    f_welch, Pxx = signal.welch(
        samples,
        fs,
        nperseg=nperseg_val,
    )

    df = f_welch[1] - f_welch[0]
    cumulative_rms = np.sqrt(np.cumsum(Pxx) * df)

    interest = f_welch > 0
    f_plot = f_welch[interest]
    Pxx_plot = Pxx[interest]
    rms_plot = cumulative_rms[interest]

    ax[0].loglog(
        f_plot,
        Pxx_plot,
        linestyle=uniform_linestyle,
        linewidth=uniform_linewidth,
        alpha=0.85,
        label=label,
    )

    ax[1].semilogx(
        f_plot,
        rms_plot,
        linestyle=uniform_linestyle,
        linewidth=uniform_linewidth,
        alpha=0.85,
        label=label,
    )

# ============================================================
# PSD formatting
# ============================================================
ax[0].set_title("Noise Spectral Density", fontweight="bold")
ax[0].set_ylabel("NSD [mV²/Hz]", fontweight="bold")
ax[0].grid(True, which="both", linestyle="--", alpha=0.6)
ax[0].legend(loc='upper left')

# ============================================================
# RMS formatting
# ============================================================
ax[1].set_title("Integrated RMS Noise", fontweight="bold")
ax[1].set_xlabel("Frequency [Hz]", fontweight="bold")
ax[1].set_ylabel("Cumulative RMS noise [mV]", fontweight="bold")
ax[1].grid(True, which="both", linestyle="--", alpha=0.6)

plt.tight_layout()
plt.show()
