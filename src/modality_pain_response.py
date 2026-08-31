"""Which physiological channel best encodes pain in BioVid Part C — and is it camera-visible?

Every dead end so far had one root cause: GSR carries pain but is invisible to a camera
(electrodermal), while HR is camera-recoverable but barely carries pain (~3.5 bpm). BioVid also
records FACIAL EMG — corrugator (brow-furrow, the textbook pain muscle) and zygomaticus (cheek).
Facial-muscle activity IS visible facial movement, so it is a camera-recoverable channel that
might also be strongly pain-linked — the combination GSR and HR each lack.

We run the identical stimulus-locked dose-response used for GSR, on all channels, all 87 subjects:
  corr(pain intensity, peak channel response), median across subjects, and % responders.

Channels: gsr (µS), hr (bpm, from ECG R-peaks), corrugator & zygomaticus EMG (rectified envelope).
This is a positive characterization: it ranks the modalities and identifies the camera-visible +
pain-predictive sweet spot.
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


def emg_envelope(x):
    """Rectified, low-pass EMG envelope (linear-envelope, standard EMG amplitude readout)."""
    b, a = signal.butter(4, [20 / (FS / 2), 250 / (FS / 2)], btype="band")
    x = signal.filtfilt(b, a, x)                       # EMG passband
    be, ae = signal.butter(2, 1.0 / (FS / 2), btype="low")
    return signal.filtfilt(be, ae, np.abs(x))          # 1 Hz linear envelope


def inst_hr(ecg):
    b, a = signal.butter(3, [5 / (FS / 2), 20 / (FS / 2)], btype="band")
    f = signal.filtfilt(b, a, ecg)
    pk, _ = signal.find_peaks(f, distance=int(FS * 0.4), height=np.std(f))
    if len(pk) < 5:
        return None
    tc = (pk[1:] + pk[:-1]) / 2
    hr = 60.0 * FS / np.diff(pk)
    return np.interp(np.arange(len(ecg)), tc, hr, left=hr[0], right=hr[-1])


def dose_response(sig_, t_us, stim, amp_min):
    n_pre, n_post = int(PRE * FS), int(POST * FS)
    ep = {l: [] for l in (1, 2, 3, 4)}
    for _, row in stim.iterrows():
        l = int(row["label"])
        if l not in ep:
            continue
        idx = int(np.searchsorted(t_us, row["time"]))
        if idx - n_pre < 0 or idx + n_post >= len(sig_):
            continue
        seg = sig_[idx - n_pre: idx + n_post] - sig_[idx - n_pre: idx].mean()
        ep[l].append(seg)
    peaks = {l: np.mean(v, 0)[n_pre:].max() for l, v in ep.items() if v}
    if len(peaks) < 2:
        return None
    labs = np.array(sorted(peaks)); vals = np.array([peaks[l] for l in labs])
    r = float(np.corrcoef(labs, vals)[0, 1])
    return r, peaks.get(4, np.nan), (r > 0.5 and peaks.get(4, 0) >= amp_min)


# camera-visibility tag + a responder amplitude floor per channel (its own units)
CHANNELS = {
    "GSR (µS)":        dict(visible="no  (electrodermal)", amp_min=0.10),
    "HR (bpm)":        dict(visible="weak (subtle color)", amp_min=1.0),
    "corrugator EMG":  dict(visible="YES (brow furrow)",   amp_min=None),
    "zygomaticus EMG": dict(visible="YES (cheek)",         amp_min=None),
}


def main():
    z = zipfile.ZipFile(BIO_ZIP)
    files = {Path(n).stem: n for n in z.namelist() if n.endswith(".csv")}
    rows = {c: [] for c in CHANNELS}
    for s in sorted(files):
        sp = STIM_DIR / f"{s}.csv"
        if not sp.exists():
            continue
        bio = pd.read_csv(io.BytesIO(z.read(files[s])), sep="\t")
        t = bio["time"].to_numpy(); stim = pd.read_csv(sp, sep="\t")
        chans = {
            "GSR (µS)": bio["gsr"].to_numpy(float),
            "HR (bpm)": inst_hr(bio["ecg"].to_numpy(float)),
            "corrugator EMG": emg_envelope(bio["emg_corrugator"].to_numpy(float)),
            "zygomaticus EMG": emg_envelope(bio["emg_zygomaticus"].to_numpy(float)),
        }
        # EMG responder floor = 20% of that subject's median envelope (relative, unit-free)
        for name, sig_ in chans.items():
            if sig_ is None:
                continue
            amp_min = CHANNELS[name]["amp_min"]
            if amp_min is None:
                amp_min = 0.2 * np.median(np.abs(sig_))
            res = dose_response(sig_, t, stim, amp_min)
            if res:
                rows[name].append(res)

    print("=" * 74)
    print("PAIN DOSE-RESPONSE BY MODALITY  (BioVid Part C, 87 subjects)")
    print("=" * 74)
    print(f"  {'channel':<16}{'median r':>9}{'ordering>0.5':>13}{'responders':>12}   camera-visible?")
    print("  " + "-" * 70)
    summ = {}
    for name in CHANNELS:
        R = np.array([r for r, _, _ in rows[name]])
        resp = np.array([b for _, _, b in rows[name]])
        summ[name] = (np.median(R), (R > 0.5).mean(), resp.mean(), len(R))
        print(f"  {name:<16}{np.median(R):>+9.3f}{100*(R>0.5).mean():>11.0f}% "
              f"{100*resp.mean():>10.0f}%   {CHANNELS[name]['visible']}")
    pd.DataFrame({k: [f"{v[0]:.3f}", f"{v[1]:.2f}", f"{v[2]:.2f}", v[3]] for k, v in summ.items()},
                 index=["median_r", "ordering_frac", "responder_frac", "n"]).to_csv(
        ROOT / "notes/modality_pain_response.csv")
    print("\n  The sweet spot = high pain response AND camera-visible.")
    print(f"  saved -> notes/modality_pain_response.csv")


if __name__ == "__main__":
    main()
