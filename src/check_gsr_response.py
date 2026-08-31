"""Validate the target signal: does GSR track pain intensity in BioVid Part C?

Stimulus-locked averaging of the GSR trace. If higher pain intensities produce
larger skin-conductance responses, the signal we want to recover from video is
real and worth recovering. If not, the whole direction dies here.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/adityaacharyaresearch/biovid-pain-project/data")
FS = 512  # Hz, biosignal sample rate
PRE, POST = 5.0, 15.0  # seconds around stimulus onset

subject = sys.argv[1] if len(sys.argv) > 1 else "071709_w_23"

bio = pd.read_csv(ROOT / f"probe_bio/biosignals_raw/{subject}.csv", sep="\t")
stim = pd.read_csv(ROOT / f"partC/stim_x/stimulus/{subject}.csv", sep="\t")

t_us = bio["time"].to_numpy()
gsr = bio["gsr"].to_numpy()
dur_s = (t_us[-1] - t_us[0]) / 1e6

print(f"subject         : {subject}")
print(f"duration        : {dur_s/60:.1f} min ({len(bio):,} samples)")
print(f"gsr range       : {gsr.min():.3f} – {gsr.max():.3f} µS")
print(f"stimuli         : {len(stim)}  intensities {sorted(stim['label'].unique())}")
print()

# stimulus-locked epochs, baseline-corrected to the pre-onset mean
n_pre, n_post = int(PRE * FS), int(POST * FS)
epochs = {lab: [] for lab in sorted(stim["label"].unique())}

for _, row in stim.iterrows():
    idx = int(np.searchsorted(t_us, row["time"]))
    if idx - n_pre < 0 or idx + n_post >= len(gsr):
        continue
    seg = gsr[idx - n_pre : idx + n_post].astype(float)
    seg = seg - seg[:n_pre].mean()  # baseline-correct
    epochs[row["label"]].append(seg)

print(f"{'intensity':>10} {'n':>4} {'peak ΔGSR (µS)':>16} {'mean ΔGSR (µS)':>16}")
print("-" * 50)
peaks = {}
for lab, segs in epochs.items():
    if not segs:
        continue
    avg = np.mean(segs, axis=0)
    resp = avg[n_pre:]  # post-onset
    peak, mean = resp.max(), resp.mean()
    peaks[lab] = peak
    print(f"{lab:>10} {len(segs):>4} {peak:>16.4f} {mean:>16.4f}")

print()
if len(peaks) > 1:
    labs = np.array(sorted(peaks))
    vals = np.array([peaks[l] for l in labs], dtype=float)
    r = np.corrcoef(labs, vals)[0, 1]
    print(f"corr(intensity, peak ΔGSR) = {r:+.3f}")
    verdict = "GSR TRACKS PAIN — target signal is real" if r > 0.5 else "WEAK/ABSENT — investigate"
    print(f"verdict: {verdict}")
