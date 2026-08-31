"""Does the POS-recovered (video) heart rate itself correlate with pain intensity?

Every earlier number was a proxy for this question, not a direct answer:
  - ECG-HR vs pain (hr_pain_response.py, 87 subjects): the CEILING -- how much pain-signal
    exists in heart rate at all, from the perfect (contact) sensor.
  - video-HR vs ECG-HR (rppg_validation.py): the RECOVERY ERROR -- how well POS reproduces
    heart rate, full stop, regardless of pain.

Neither directly tests "does POS-recovered HR track pain intensity." This does that directly:
same stimulus-locked dose-response method as every other channel (baseline-correct each
pain-onset epoch, take the post-onset peak, correlate peak vs. intensity 1-4), but applied to
the POS video-HR time series instead of GSR/ECG/EMG. Only the 3 subjects with cached video
traces (n=3, not the 87-subject power of the ECG ceiling) -- but it's the real, direct number.

Caveat worth stating: each POS-HR estimate already averages a 15s sliding window (WIN_S in
rppg_validation.py), so a fast pain-onset HR change is partly smoothed out before we even get
to the dose-response step. That's compounded on top of the recovery noise itself.
"""
import glob
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

np.seterr(all="ignore")
ROOT = Path("/Users/adityaacharyaresearch/biovid-pain-project")
spec = importlib.util.spec_from_file_location("rv", ROOT / "src/rppg_validation.py")
rv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rv)

PRE, POST = 5.0, 15.0   # seconds around stimulus onset, matching the other dose-response checks


def dose_response(t, hr, stim):
    """Same baseline-correct + post-peak + correlate design as every other channel here."""
    ep = {l: [] for l in (1, 2, 3, 4)}
    t0s = stim["time"].to_numpy() / 1e6
    labs = stim["label"].to_numpy()
    for t0, lab in zip(t0s, labs):
        lab = int(lab)
        if lab not in ep:
            continue
        pre = hr[(t >= t0 - PRE) & (t < t0)]
        post_mask = (t >= t0) & (t < t0 + POST)
        if len(pre) < 1 or post_mask.sum() < 1:
            continue
        base = np.nanmean(pre)
        if not np.isfinite(base):
            continue
        peak = np.nanmax(hr[post_mask]) - base
        if np.isfinite(peak):
            ep[lab].append(peak)
    peaks = {l: np.mean(v) for l, v in ep.items() if v}
    if len(peaks) < 2:
        return None
    ls = np.array(sorted(peaks)); vs = np.array([peaks[l] for l in ls])
    return float(np.corrcoef(ls, vs)[0, 1]), peaks, {l: len(ep[l]) for l in ep}


def main():
    traces = sorted(glob.glob(str(ROOT / "notes/trace_*_s1.npz")))
    print("=" * 74)
    print("PAIN DOSE-RESPONSE OF POS-RECOVERED (VIDEO) HEART RATE  vs.  ECG HEART RATE")
    print("=" * 74)
    print(f"  {'subject':14}{'video-HR r':>12}{'ECG-HR r':>11}   (same subject, same epochs)")
    print("  " + "-" * 55)

    vid_rs, ecg_rs = [], []
    for tf in traces:
        s = tf.split("trace_")[1].rsplit("_s1", 1)[0]
        d = np.load(tf)
        stim = pd.read_csv(ROOT / f"data/partC/stim_x/stimulus/{s}.csv", sep="\t")
        bio = pd.read_csv(ROOT / f"data/probe_bio/biosignals_raw/{s}.csv", sep="\t",
                          usecols=["time", "ecg"])

        # video-HR via POS, same pipeline used for the accuracy validation
        pulse = rv.pos_pulse(d["rgb"], float(d["fps"]))
        tv, hrv, _ = rv.video_hr(pulse, float(d["fps"]))
        res_v = dose_response(tv, hrv, stim)

        # ECG-HR on THIS subject, resampled onto the SAME timestamps as video-HR (fair epochs)
        ecg = bio["ecg"].to_numpy(float)
        from scipy import signal
        b, a = signal.butter(3, [5 / (rv.FS_ECG / 2), 20 / (rv.FS_ECG / 2)], btype="band")
        f = signal.filtfilt(b, a, ecg)
        pk, _ = signal.find_peaks(f, distance=int(rv.FS_ECG * 0.4), height=np.std(f))
        tpk = bio["time"].to_numpy()[pk] / 1e6
        ihr = 60.0 / np.diff(tpk)
        tmid = (tpk[1:] + tpk[:-1]) / 2
        te = tv  # same timestamps as video-HR windows, for an apples-to-apples epoch comparison
        hre = np.interp(te, tmid, ihr, left=ihr[0], right=ihr[-1])
        res_e = dose_response(te, hre, stim)

        rv_, re_ = (res_v[0] if res_v else np.nan), (res_e[0] if res_e else np.nan)
        vid_rs.append(rv_); ecg_rs.append(re_)
        print(f"  {s:14}{rv_:>+12.3f}{re_:>+11.3f}")

    print(f"\n  median video-HR r = {np.nanmedian(vid_rs):+.3f}   (n={len(vid_rs)} subjects)")
    print(f"  median ECG-HR   r = {np.nanmedian(ecg_rs):+.3f}   (same subjects, same epochs)")
    print("\n  Interpretation: ECG-HR's own correlation is the ceiling on these 3 subjects;")
    print("  video-HR's correlation is what survives after POS recovery. The gap between them")
    print("  is the cost of imperfect video recovery, isolated from the 87-subject ceiling study.")


if __name__ == "__main__":
    main()
