"""Positive control: can the probe recover the CARDIAC signal (pulse rate) from video?

The EDA-from-video result is a null. That is only interpretable if the same pipeline CAN
recover a signal we know is present in face video: the heart rate. We compare the video POS
rPPG pulse rate against the ground-truth heart rate from the ECG in the same biosignal file,
window for window.

  * probe recovers HR (video-HR vs ECG-HR correlates)  -> pipeline works; the EDA null is a
    real physiological/regime finding ("cardiac recoverable at 25 Hz, electrodermal not").
  * probe recovers neither                              -> probe/regime too weak; null is moot.
"""
import glob
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import spearmanr

np.seterr(all="ignore")
ROOT = Path("/Users/adityaacharyaresearch/biovid-pain-project")
spec = importlib.util.spec_from_file_location("probe", ROOT / "src/probe_eda_from_video.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
FS_BIO = 512


def ecg_hr(subject, centres):
    """Ground-truth HR (bpm) per 60 s window from ECG R-peaks."""
    bio = pd.read_csv(ROOT / f"data/probe_bio/biosignals_raw/{subject}.csv", sep="\t",
                      usecols=["time", "ecg"])
    t = bio["time"].to_numpy() / 1e6
    ecg = bio["ecg"].to_numpy(float)
    b, a = signal.butter(3, [5 / (FS_BIO / 2), 20 / (FS_BIO / 2)], btype="band")
    f = signal.filtfilt(b, a, ecg)
    pk, _ = signal.find_peaks(f, distance=int(FS_BIO * 0.4), height=np.std(f))
    tp = t[pk]
    hr = []
    for tc in centres:
        m = (tp >= tc - probe.WIN_S / 2) & (tp < tc + probe.WIN_S / 2)
        hr.append(60.0 / np.median(np.diff(tp[m])) if m.sum() > 3 else np.nan)
    return np.array(hr)


def main():
    subjects = sorted(p.split("trace_")[1].rsplit("_s", 1)[0]
                      for p in glob.glob(str(ROOT / "notes/trace_*_s2.npz")))
    print(f"{'subject':14} {'n':>4} {'video-HR vs ECG-HR r':>22} {'p':>10}   verdict")
    print("-" * 70)
    rs = []
    for s in subjects:
        d = np.load(ROOT / f"notes/trace_{s}_s2.npz")
        t, feats, _ = probe.compute_features(d["rgb"], d["geom"], float(d["fps"]))
        vid_hr = -feats["pulse_rate"]                 # probe stored HR as -60/ibi
        ref_hr = ecg_hr(s, t)
        m = np.isfinite(vid_hr) & np.isfinite(ref_hr)
        r, p = spearmanr(vid_hr[m], ref_hr[m])
        rs.append(r)
        flag = "recovers HR" if (r > 0.3 and p < 0.05) else "no"
        print(f"{s:14} {m.sum():>4} {r:>+22.3f} {p:>10.4f}   {flag}")
        # also report the ECG-HR and video-HR medians as a sanity check on plausibility
        print(f"{'':14}      ECG-HR med {np.nanmedian(ref_hr):5.1f} bpm | "
              f"video-HR med {np.nanmedian(vid_hr):5.1f} bpm")

    print("\n" + "=" * 60)
    good = sum(1 for r in rs if r > 0.3)
    print(f"  subjects where video recovers HR (r>0.3): {good}/{len(rs)}")
    if good >= 2:
        print("  => PIPELINE WORKS on the easy (cardiac) signal.")
        print("     The EDA null is therefore a real result: at 25 Hz on freely-moving")
        print("     BioVid faces, the cardiac signal is recoverable but the electrodermal")
        print("     signal is not — it is lost to motion-arousal confounding.")
    else:
        print("  => Probe fails even on the cardiac signal. The EDA null is NOT")
        print("     interpretable yet; strengthen rPPG (CHROM/POS tuning, ROI, detrend)")
        print("     before concluding anything about EDA.")


if __name__ == "__main__":
    main()
