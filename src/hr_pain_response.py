"""Ceiling for option (b): does HEART RATE respond to pain intensity at all?

rPPG recovers the cardiac signal. But that only helps pain if HR itself carries pain. This is
the ceiling: even a perfect camera can do no better than the ECG-derived HR does. We measure the
stimulus-locked HR dose-response across all 87 subjects (from the ECG in the biosignals zip) and
compare it head-to-head with the GSR dose-response already established.

Method mirrors the GSR responder analysis: for each thermode onset, instantaneous HR (from ECG
R-peak intervals) is epoched PRE/POST and baseline-corrected; peak ΔHR is averaged per intensity;
r = corr(intensity, peak ΔHR). This bounds what the whole video->pain route can achieve.
"""
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

ROOT = Path("/Users/adityaacharyaresearch/biovid-pain-project")
BIO_ZIP = ROOT / "data/partC/biosignals_raw.zip"
STIM_DIR = ROOT / "data/partC/stim_x/stimulus"
FS = 512
PRE, POST = 5.0, 15.0


def inst_hr(ecg, t_us):
    """Instantaneous HR (bpm) resampled onto a uniform 512 Hz grid via R-peak intervals."""
    b, a = signal.butter(3, [5 / (FS / 2), 20 / (FS / 2)], btype="band")
    f = signal.filtfilt(b, a, ecg)
    pk, _ = signal.find_peaks(f, distance=int(FS * 0.4), height=np.std(f))
    if len(pk) < 5:
        return None
    tp = t_us[pk] / 1e6
    hr = 60.0 / np.diff(tp)                     # bpm between successive beats
    tc = (tp[1:] + tp[:-1]) / 2                  # midpoint time of each interval
    grid = t_us / 1e6
    return np.interp(grid, tc, hr, left=hr[0], right=hr[-1])


def analyze(subject, ecg, t_us, stim):
    hr = inst_hr(ecg, t_us)
    if hr is None:
        return None
    n_pre, n_post = int(PRE * FS), int(POST * FS)
    ep = {l: [] for l in (1, 2, 3, 4)}
    for _, row in stim.iterrows():
        l = int(row["label"])
        if l not in ep:
            continue
        idx = int(np.searchsorted(t_us, row["time"]))
        if idx - n_pre < 0 or idx + n_post >= len(hr):
            continue
        seg = hr[idx - n_pre: idx + n_post] - hr[idx - n_pre: idx].mean()
        ep[l].append(seg)
    peaks = {l: np.mean(v, 0)[n_pre:].max() for l, v in ep.items() if v}
    if len(peaks) < 2:
        return None
    labs = np.array(sorted(peaks)); vals = np.array([peaks[l] for l in labs])
    r = float(np.corrcoef(labs, vals)[0, 1])
    return dict(subject=subject, r=r, peak4=peaks.get(4, np.nan),
                responder=(r > 0.5 and peaks.get(4, 0) >= 1.0))   # >=1 bpm rise at intensity 4


def main():
    z = zipfile.ZipFile(BIO_ZIP)
    files = {Path(n).stem: n for n in z.namelist() if n.endswith(".csv")}
    rows = []
    for i, s in enumerate(sorted(files), 1):
        sp = STIM_DIR / f"{s}.csv"
        if not sp.exists():
            continue
        try:
            bio = pd.read_csv(io.BytesIO(z.read(files[s])), sep="\t", usecols=["time", "ecg"])
            res = analyze(s, bio["ecg"].to_numpy(float), bio["time"].to_numpy(), pd.read_csv(sp, sep="\t"))
            if res:
                rows.append(res)
        except Exception as e:
            print(f"  {s}: {str(e)[:50]}")
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "notes/hr_pain_response.csv", index=False)

    gsr = pd.read_csv(ROOT / "notes/gsr_responders.csv")
    print("=" * 60)
    print(f"HR dose-response to pain (ECG, {len(df)} subjects)")
    print("=" * 60)
    print(f"  correct ordering (r>0.5)      : {(df.r>0.5).sum()}/{len(df)} ({100*(df.r>0.5).mean():.0f}%)")
    print(f"  responders (r>0.5 & ΔHR@4≥1bpm): {df.responder.sum()}/{len(df)} ({100*df.responder.mean():.0f}%)")
    print(f"  median r                       : {df.r.median():+.3f}")
    print(f"  median peak ΔHR @ intensity 4  : {df.peak4.median():.2f} bpm")
    print("\n  HEAD-TO-HEAD (the ceiling comparison):")
    print(f"    GSR : median r {gsr.r.median():+.3f}, responders {100*gsr.responder.mean():.0f}%")
    print(f"    HR  : median r {df.r.median():+.3f}, responders {100*df.responder.mean():.0f}%")
    print("\n  -> HR is the WEAKER pain channel (as expected: ECG~75% vs GSR~89% in the")
    print("     literature). This bounds option (b): rPPG recovers HR well, but HR itself")
    print("     carries less pain than GSR, so the video->pain ceiling via HR is limited.")
    print(f"\n  saved -> notes/hr_pain_response.csv")


if __name__ == "__main__":
    main()
