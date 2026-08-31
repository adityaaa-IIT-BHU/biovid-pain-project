"""Scale the video-based pain-correlation checks from n=1-3 to n=10 subjects.

For each subject: download video if missing, extract biosignals if missing, extract a
full-rate forehead trace (cached), then run two independent checks against the SAME
stimulus-locked dose-response design used everywhere else in this project:

  1. POS route: video heart rate (via POS rPPG) vs ECG heart rate (accuracy), and
     video heart rate vs pain intensity (does the recovered signal itself track pain).
  2. Facial route: py-feat Action Units -> PSPI / AU04 alone vs pain intensity.

Every subject's result is appended to CSV as soon as it's computed, so a crash or disk
issue partway through does not lose what's already done (the CARE-PD run lost everything
by only printing to stdout -- this project does not repeat that mistake).

Detectorv1 (py-feat) is loaded ONCE for the whole run, not per subject (~90s each time).
"""
import io
import os
import struct
import sys
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import requests
from remotezip import RemoteZip
from scipy import signal

ROOT = Path("/Users/adityaacharyaresearch/biovid-pain-project")
BASE = "https://cloud.ovgu.de/public.php/webdav/PartC"
AUTH = (os.environ["BIOVID_TOKEN"], os.environ["BIOVID_PASSWORD"])  # rotates; email sascha.gruss@uni-ulm.de if it stops working
BIO_ZIP = ROOT / "data/partC/biosignals_raw.zip"
STIM_DIR = ROOT / "data/partC/stim_x/stimulus"
CHUNK = 4 * 1024 * 1024
FS_ECG = 512
PRE_BIO, POST_BIO = 5.0, 15.0     # pain-epoch window for POS-HR / ECG-HR dose-response
PRE_AU, POST_AU, STEP_AU = 4.0, 14.0, 2.0   # sparser window for the slow py-feat AU pass

EXISTING = ["071709_w_23", "080314_w_25", "080709_m_24"]
NEW = ["071309_w_21", "071313_m_41", "071614_m_20", "071814_w_23",
       "071911_w_24", "072414_m_23", "072514_m_27"]
SUBJECTS = EXISTING + NEW

sys.path.insert(0, str(ROOT / "src"))


# --------------------------------------------------------------------- download ----
def data_start(sess, url, header_offset):
    r = sess.get(url, headers={"Range": f"bytes={header_offset}-{header_offset+29}"}, timeout=30)
    r.raise_for_status()
    name_len, extra_len = struct.unpack("<HH", r.content[26:30])
    return header_offset + 30 + name_len + extra_len


def ensure_video(subject):
    out = ROOT / f"data/video/{subject}.mp4"
    if out.exists() and out.stat().st_size > 1_000_000:
        return out
    sess = requests.Session(); sess.auth = AUTH
    url = f"{BASE}/video.zip"
    with RemoteZip(url, session=sess) as z:
        info = [x for x in z.infolist() if f"{subject}.mp4" in x.filename][0]
        assert info.compress_type == 0
        total = info.file_size
        start = data_start(sess, url, info.header_offset)
    out.parent.mkdir(parents=True, exist_ok=True)
    have = out.stat().st_size if out.exists() else 0
    print(f"    downloading {subject}: {total/1e6:.0f} MB", flush=True)
    attempt = 0
    while have < total:
        lo, hi = start + have, min(start + have + CHUNK, start + total) - 1
        try:
            r = sess.get(url, headers={"Range": f"bytes={lo}-{hi}"}, stream=True, timeout=(15, 30))
            r.raise_for_status()
            with open(out, "ab") as f:
                for block in r.iter_content(256 * 1024):
                    if block:
                        f.write(block)
            have = out.stat().st_size
            attempt = 0
        except Exception as e:
            attempt += 1
            time.sleep(min(30, 2 ** attempt))
            have = out.stat().st_size
    print(f"    {subject} video done ({have/1e6:.0f} MB)", flush=True)
    return out


def ensure_biosignals(subject):
    out = ROOT / f"data/probe_bio/biosignals_raw/{subject}.csv"
    if out.exists():
        return out
    z = zipfile.ZipFile(BIO_ZIP)
    name = [n for n in z.namelist() if f"{subject}.csv" in n][0]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(z.read(name))
    return out


def ensure_trace(subject, extract_forehead):
    cache = ROOT / f"notes/trace_{subject}_s1.npz"
    if cache.exists():
        d = np.load(cache)
        return d["rgb"], d["geom"], float(d["fps"])
    vid = ensure_video(subject)
    print(f"    extracting forehead trace for {subject}...", flush=True)
    t0 = time.time()
    rgb, geom, fps, ok = extract_forehead(vid, None, 1)
    np.savez_compressed(cache, rgb=rgb, geom=geom, fps=fps, ok=ok)
    print(f"    trace done in {(time.time()-t0)/60:.1f} min ({ok}/{len(rgb)} frames w/ face)",
          flush=True)
    return rgb, geom, fps


# ---------------------------------------------------------- dose-response (shared) ----
def dose_response(t, sig_, stim, pre, post):
    ep = {l: [] for l in (1, 2, 3, 4)}
    for t0, lab in zip(stim["time"].to_numpy() / 1e6, stim["label"].to_numpy()):
        lab = int(lab)
        if lab not in ep:
            continue
        pre_v = sig_[(t >= t0 - pre) & (t < t0)]
        post_mask = (t >= t0) & (t < t0 + post)
        if len(pre_v) < 1 or post_mask.sum() < 1:
            continue
        base = np.nanmean(pre_v)
        if not np.isfinite(base):
            continue
        peak = np.nanmax(sig_[post_mask]) - base
        if np.isfinite(peak):
            ep[lab].append(peak)
    peaks = {l: np.mean(v) for l, v in ep.items() if v}
    if len(peaks) < 2:
        return np.nan
    ls = np.array(sorted(peaks)); vs = np.array([peaks[l] for l in ls])
    return float(np.corrcoef(ls, vs)[0, 1])


def ecg_hr_series(subject, t_query):
    bio = pd.read_csv(ROOT / f"data/probe_bio/biosignals_raw/{subject}.csv", sep="\t",
                      usecols=["time", "ecg"])
    b, a = signal.butter(3, [5 / (FS_ECG / 2), 20 / (FS_ECG / 2)], btype="band")
    f = signal.filtfilt(b, a, bio["ecg"].to_numpy(float))
    pk, _ = signal.find_peaks(f, distance=int(FS_ECG * 0.4), height=np.std(f))
    tpk = bio["time"].to_numpy()[pk] / 1e6
    ihr = 60.0 / np.diff(tpk)
    tmid = (tpk[1:] + tpk[:-1]) / 2
    return np.interp(t_query, tmid, ihr, left=ihr[0], right=ihr[-1])


# ------------------------------------------------------------------------- main ----
def main():
    from probe_eda_from_video import extract_forehead
    import rppg_validation as rv
    from feat import Detectorv1

    rppg_csv = ROOT / "notes/scale10_rppg.csv"
    pspi_csv = ROOT / "notes/scale10_pspi.csv"
    if not rppg_csv.exists():
        pd.DataFrame(columns=["subject", "acc_mae_bpm", "acc_r", "vidhr_pain_r",
                              "ecghr_pain_r"]).to_csv(rppg_csv, index=False)
    if not pspi_csv.exists():
        pd.DataFrame(columns=["subject", "pspi_pain_r", "au04_pain_r", "n_epochs"]).to_csv(
            pspi_csv, index=False)
    done_rppg = set(pd.read_csv(rppg_csv)["subject"])
    done_pspi = set(pd.read_csv(pspi_csv)["subject"])

    print("loading py-feat Detector (once for the whole run)...", flush=True)
    det = Detectorv1(device="cpu")

    for si, subject in enumerate(SUBJECTS, 1):
        print(f"\n=== [{si}/{len(SUBJECTS)}] {subject} ===", flush=True)
        ensure_biosignals(subject)
        stim = pd.read_csv(STIM_DIR / f"{subject}.csv", sep="\t")
        rgb, geom, fps = ensure_trace(subject, extract_forehead)

        # --- 1. POS route ---
        if subject not in done_rppg:
            pulse = rv.pos_pulse(rgb, fps)
            tv, hrv, _ = rv.video_hr(pulse, fps)
            ehr = rv.ecg_hr(subject, tv)
            m = np.isfinite(hrv) & np.isfinite(ehr)
            acc_mae = float(np.mean(np.abs(hrv[m] - ehr[m]))) if m.sum() else np.nan
            acc_r = float(np.corrcoef(hrv[m], ehr[m])[0, 1]) if m.sum() > 2 else np.nan
            vidhr_r = dose_response(tv, hrv, stim, PRE_BIO, POST_BIO)
            ecg_series = ecg_hr_series(subject, tv)
            ecghr_r = dose_response(tv, ecg_series, stim, PRE_BIO, POST_BIO)
            pd.DataFrame([dict(subject=subject, acc_mae_bpm=acc_mae, acc_r=acc_r,
                               vidhr_pain_r=vidhr_r, ecghr_pain_r=ecghr_r)]).to_csv(
                rppg_csv, mode="a", header=False, index=False)
            print(f"    POS: MAE={acc_mae:.1f}bpm  acc_r={acc_r:+.2f}  "
                  f"vidHR-vs-pain r={vidhr_r:+.2f}  ECG-HR-vs-pain r={ecghr_r:+.2f}", flush=True)
        else:
            print("    POS: already done, skipping", flush=True)

        # --- 2. facial AU / PSPI route ---
        if subject not in done_pspi:
            import tempfile, shutil
            offsets = np.arange(-PRE_AU, POST_AU + 1e-6, STEP_AU)
            cap = cv2.VideoCapture(str(ROOT / f"data/video/{subject}.mp4"))
            tmp = Path(tempfile.mkdtemp())
            paths, meta = [], []
            for sidx, row in stim.iterrows():
                t0 = row["time"] / 1e6
                for off in offsets:
                    fidx = int(round((t0 + off) * fps))
                    cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
                    ret, frame = cap.read()
                    if not ret:
                        continue
                    p = tmp / f"s{sidx}_o{off:+.1f}.jpg"
                    cv2.imwrite(str(p), frame)
                    paths.append(str(p)); meta.append((sidx, int(row["label"]), off))
            cap.release()
            print(f"    AU: {len(paths)} frames, running detector "
                  f"(~{len(paths)*1.2/60:.0f} min)...", flush=True)
            t0 = time.time()
            out = det.detect(paths, data_type="image", progress_bar=False).reset_index(drop=True)
            out["stim_idx"] = [m[0] for m in meta]; out["intensity"] = [m[1] for m in meta]
            out["offset"] = [m[2] for m in meta]
            out["PSPI"] = (out["AU04"] + out[["AU06", "AU07"]].max(axis=1)
                          + out[["AU09", "AU10"]].max(axis=1) + out["AU43"])
            shutil.rmtree(tmp, ignore_errors=True)

            rows = []
            for sidx, g in out.groupby("stim_idx"):
                pre_g, post_g = g[g.offset < 0], g[g.offset >= 0]
                if len(pre_g) < 1 or len(post_g) < 1:
                    continue
                rows.append(dict(intensity=g["intensity"].iloc[0],
                                 peak_pspi=post_g["PSPI"].max() - pre_g["PSPI"].mean(),
                                 peak_au4=post_g["AU04"].max() - pre_g["AU04"].mean()))
            ep = pd.DataFrame(rows)
            by_int_p = ep.groupby("intensity")["peak_pspi"].mean()
            by_int_a = ep.groupby("intensity")["peak_au4"].mean()
            pspi_r = float(np.corrcoef(by_int_p.index, by_int_p.values)[0, 1]) if len(by_int_p) > 1 else np.nan
            au04_r = float(np.corrcoef(by_int_a.index, by_int_a.values)[0, 1]) if len(by_int_a) > 1 else np.nan
            pd.DataFrame([dict(subject=subject, pspi_pain_r=pspi_r, au04_pain_r=au04_r,
                               n_epochs=len(ep))]).to_csv(pspi_csv, mode="a", header=False, index=False)
            print(f"    AU done in {(time.time()-t0)/60:.1f} min: "
                  f"PSPI r={pspi_r:+.2f}  AU04 r={au04_r:+.2f}", flush=True)
        else:
            print("    AU: already done, skipping", flush=True)

    print("\n" + "=" * 70)
    print("ALL SUBJECTS DONE")
    print("=" * 70)
    rdf, pdf = pd.read_csv(rppg_csv), pd.read_csv(pspi_csv)
    print(rdf.to_string(index=False))
    print(pdf.to_string(index=False))
    print(f"\nmedian vidHR-vs-pain r  = {rdf['vidhr_pain_r'].median():+.3f}")
    print(f"median ECG-HR-vs-pain r = {rdf['ecghr_pain_r'].median():+.3f}")
    print(f"median PSPI-vs-pain r   = {pdf['pspi_pain_r'].median():+.3f}")
    print(f"median AU04-vs-pain r   = {pdf['au04_pain_r'].median():+.3f}")


if __name__ == "__main__":
    main()
