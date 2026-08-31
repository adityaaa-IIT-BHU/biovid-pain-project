"""rPPG done right: validate video-recovered pulse against the in-dataset ECG.

My earlier "positive control" reported one Spearman r on 60 s windows. That is not how rPPG is
validated. The field standard (ISO 80601-2-61 style, and every rPPG paper) reports, on short
sliding windows, agreement between the camera HR and a gold-standard HR:

  * HR estimate per window from the POS pulse spectrum (PSD peak in the cardiac band), which is
    the standard rPPG HR estimator, plus a signal-quality SNR.
  * ECG reference HR per matched window from R-peak intervals.
  * agreement metrics: Pearson r, MAE, RMSE, MAPE, % of windows within 5 bpm (the accepted
    "good" threshold), and a Bland-Altman analysis (bias + 95% limits of agreement) which,
    unlike a correlation, exposes systematic offset and how tightly the two track beat-rate.

Windows: 15 s (long enough to resolve a cardiac peak, short enough to track HR changes), 1 s hop.

Usage:  python3 src/rppg_validation.py [--stride 1|2]
Outputs: notes/rppg_metrics.csv, notes/figures/{scatter,bland_altman}.png
"""
import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.seterr(all="ignore")
ROOT = Path("/Users/adityaacharyaresearch/biovid-pain-project")
FS_ECG = 512
HR_LO, HR_HI = 0.7, 3.3          # Hz  -> 42-198 bpm cardiac band
WIN_S, STEP_S = 15.0, 1.0


def pos_pulse(rgb, fps):
    """POS (Wang 2017) overlap-add rPPG from forehead RGB."""
    from numpy.linalg import norm
    C = np.vstack([np.interp(np.arange(len(rgb)),
                             np.where(np.isfinite(rgb[:, k]))[0],
                             rgb[np.isfinite(rgb[:, k]), k]) for k in range(3)]).T
    N = len(C)
    l = max(8, int(fps * 1.6))
    H = np.zeros(N)
    for n in range(0, N - l):
        Cn = C[n:n + l] / (C[n:n + l].mean(0) + 1e-8)
        S1 = Cn[:, 1] - Cn[:, 2]
        S2 = -2 * Cn[:, 0] + Cn[:, 1] + Cn[:, 2]
        h = S1 + (S1.std() / (S2.std() + 1e-8)) * S2
        H[n:n + l] += h - h.mean()
    b, a = signal.butter(3, [HR_LO / (fps / 2), min(HR_HI / (fps / 2), 0.99)], btype="band")
    return signal.filtfilt(b, a, H)


def video_hr(pulse, fps):
    """Per-window HR (bpm) from the PSD peak, and a band SNR quality metric."""
    win, step = int(WIN_S * fps), int(STEP_S * fps)
    t, hr, snr = [], [], []
    for s in range(0, len(pulse) - win, step):
        seg = pulse[s:s + win]
        f, p = signal.welch(seg, fs=fps, nperseg=min(len(seg), int(fps * 8)))
        band = (f >= HR_LO) & (f <= HR_HI)
        if band.sum() < 3:
            continue
        fb, pb = f[band], p[band]
        fpk = fb[np.argmax(pb)]
        # SNR: power within +-0.1 Hz of peak & its first harmonic vs rest of band
        sig = (np.abs(fb - fpk) < 0.1) | (np.abs(fb - 2 * fpk) < 0.1)
        snr.append(10 * np.log10(pb[sig].sum() / (pb[~sig].sum() + 1e-12)))
        hr.append(60 * fpk)
        t.append((s + win / 2) / fps)
    return np.array(t), np.array(hr), np.array(snr)


def ecg_hr(subject, centres):
    bio = pd.read_csv(ROOT / f"data/probe_bio/biosignals_raw/{subject}.csv", sep="\t",
                      usecols=["time", "ecg"])
    tb = bio["time"].to_numpy() / 1e6
    b, a = signal.butter(3, [5 / (FS_ECG / 2), 20 / (FS_ECG / 2)], btype="band")
    f = signal.filtfilt(b, a, bio["ecg"].to_numpy(float))
    pk, _ = signal.find_peaks(f, distance=int(FS_ECG * 0.4), height=np.std(f))
    tp = tb[pk]
    hr = []
    for tc in centres:
        m = (tp >= tc - WIN_S / 2) & (tp < tc + WIN_S / 2)
        hr.append(60.0 / np.median(np.diff(tp[m])) if m.sum() > 3 else np.nan)
    return np.array(hr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=2)
    a = ap.parse_args()
    traces = sorted(glob.glob(str(ROOT / f"notes/trace_*_s{a.stride}.npz")))
    (ROOT / "notes/figures").mkdir(exist_ok=True)

    allv, alle, rows = [], [], []
    colors = plt.cm.tab10(np.linspace(0, 1, len(traces)))
    fig1, ax1 = plt.subplots(figsize=(5, 5))
    for i, tf in enumerate(traces):
        s = tf.split("trace_")[1].rsplit(f"_s{a.stride}", 1)[0]
        d = np.load(tf)
        pulse = pos_pulse(d["rgb"], float(d["fps"]))
        t, vhr, snr = video_hr(pulse, float(d["fps"]))
        ehr = ecg_hr(s, t)
        m = np.isfinite(vhr) & np.isfinite(ehr)
        v, e = vhr[m], ehr[m]
        allv.append(v); alle.append(e)
        r = np.corrcoef(v, e)[0, 1]
        mae = np.mean(np.abs(v - e)); rmse = np.sqrt(np.mean((v - e) ** 2))
        w5 = 100 * np.mean(np.abs(v - e) <= 5)
        rows.append(dict(subject=s, n=int(m.sum()), pearson_r=r, mae_bpm=mae, rmse_bpm=rmse,
                         within5_pct=w5, mean_snr_db=np.mean(snr[m])))
        ax1.scatter(e, v, s=6, alpha=0.4, color=colors[i], label=f"{s} (r={r:.2f})")
        print(f"  {s}: n={m.sum():3d}  r={r:+.3f}  MAE={mae:4.1f}  RMSE={rmse:4.1f}  "
              f"within5={w5:3.0f}%  SNR={np.mean(snr[m]):+.1f}dB")

    V, E = np.concatenate(allv), np.concatenate(alle)
    r = np.corrcoef(V, E)[0, 1]; mae = np.mean(np.abs(V - E)); rmse = np.sqrt(np.mean((V - E) ** 2))
    w5 = 100 * np.mean(np.abs(V - E) <= 5); mape = 100 * np.mean(np.abs(V - E) / E)
    bias = np.mean(V - E); loa = 1.96 * np.std(V - E)
    print("\n" + "=" * 60)
    print(f"POOLED (n={len(V)} windows, {len(traces)} subjects, stride {a.stride})")
    print("=" * 60)
    print(f"  Pearson r     : {r:+.3f}")
    print(f"  MAE           : {mae:.2f} bpm")
    print(f"  RMSE          : {rmse:.2f} bpm")
    print(f"  MAPE          : {mape:.2f} %")
    print(f"  within 5 bpm  : {w5:.1f} %")
    print(f"  Bland-Altman  : bias {bias:+.2f} bpm, 95% LoA [{bias-loa:+.1f}, {bias+loa:+.1f}]")

    lim = [min(E.min(), V.min()) - 5, max(E.max(), V.max()) + 5]
    ax1.plot(lim, lim, "k--", lw=1, label="identity")
    ax1.set(xlabel="ECG HR (bpm)", ylabel="video HR (bpm)", xlim=lim, ylim=lim,
            title=f"rPPG vs ECG heart rate (r={r:.2f}, MAE={mae:.1f} bpm)")
    ax1.legend(fontsize=7); fig1.tight_layout(); fig1.savefig(ROOT / "notes/figures/scatter.png", dpi=130)

    fig2, ax2 = plt.subplots(figsize=(5.5, 4))
    mean_hr = (V + E) / 2
    ax2.scatter(mean_hr, V - E, s=6, alpha=0.4)
    for y, ls, lab in [(bias, "-", f"bias {bias:+.1f}"),
                       (bias + loa, "--", f"+1.96SD {bias+loa:+.1f}"),
                       (bias - loa, "--", f"-1.96SD {bias-loa:+.1f}")]:
        ax2.axhline(y, ls=ls, color="crimson", lw=1); ax2.text(lim[1]-2, y, lab, fontsize=7, va="bottom", ha="right")
    ax2.set(xlabel="mean of video & ECG HR (bpm)", ylabel="video − ECG (bpm)",
            title="Bland–Altman: rPPG vs ECG")
    fig2.tight_layout(); fig2.savefig(ROOT / "notes/figures/bland_altman.png", dpi=130)

    pd.DataFrame(rows).to_csv(ROOT / "notes/rppg_metrics.csv", index=False)
    print(f"\n  saved -> notes/rppg_metrics.csv, notes/figures/scatter.png, bland_altman.png")


if __name__ == "__main__":
    main()
