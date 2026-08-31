"""Feasibility probe: can sympathetic arousal (GSR) be recovered from BioVid face video?

Adapts the ETH SIPLab peripheral-blood-flow approach (Spearman r~0.6 vs contact EDA at
100 Hz, chin-rested, forehead+palm) to BioVid Part C's harder regime: 25 Hz, frontal face
only, freely-moving subjects.

Design goal: a FAIR test — one that neither hides a real effect nor invents a fake one.
Two failure modes are guarded explicitly:

  * FALSE POSITIVE. On a freely-moving subject the slow (tonic) forehead-brightness signal
    is driven by head motion, lighting drift and the thermode's heat as much as by blood
    flow — and GSR drifts slowly too. So a raw correlation can be pure confound. We record a
    motion channel (forehead ROI displacement + scale change) and the thermode temperature,
    and report the PARTIAL correlation of each feature vs GSR controlling for {motion,
    temperature, time}. A signal that survives that is blood flow; one that vanishes was not.
    Overlapping windows also inflate significance, so p is computed on NON-overlapping windows.

  * FALSE NEGATIVE. Raw green-mean is the most motion-sensitive rPPG choice. The pulsatile
    features use POS (Wang et al. 2017), the standard motion-robust projection, giving the
    cardiac channel a fair shot at 25 Hz under movement.

Three candidate signals, each windowed and correlated (raw + partial) against ground-truth GSR:
  1. blood_volume - tonic component of normalized green   (ETH "total blood volume change")
  2. pulse_amp    - POS rPPG envelope amplitude            (ETH "mean blood pulsation amplitude")
  3. pulse_rate   - POS instantaneous rate, inverted       (ETH "instantaneous pulse rate")

Usage:
    python3 src/probe_eda_from_video.py <subject> [--max-frames N] [--stride K]

A partial correlation that survives the confound controls means the direction is live.
A real null at 25 Hz is itself a publishable negative -> pivots us to the rPPG fallback.
"""
import argparse
import io
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import spearmanr, rankdata

# Apple Silicon's Accelerate BLAS raises spurious over/underflow FP flags in matmul
# (inside scipy detrend/filtfilt and our residualization) that do NOT affect results:
# every feature array comes out finite and the partial correlations match an independent
# precision-matrix computation to 3 dp. Suppress the cosmetic warnings.
np.seterr(over="ignore", invalid="ignore", divide="ignore")

ROOT = Path("/Users/adityaacharyaresearch/biovid-pain-project")
DATA = ROOT / "data"
WIN_S = 60.0          # ETH used 60 s windows
STEP_S = 5.0          # window hop (dense point estimate; p uses non-overlapping subset)

# MediaPipe FaceMesh landmarks bounding the forehead (ETH: forehead is the best facial site)
LM_TOP, LM_BOTTOM, LM_LEFT, LM_RIGHT = 10, 9, 67, 297


def extract_forehead(video_path, max_frames=None, stride=1):
    """Per-frame forehead mean RGB and ROI geometry (cx, cy, scale) for a motion proxy.

    mediapipe 0.10.35 dropped the legacy `solutions` API, so this uses the Tasks
    FaceLandmarker (VIDEO mode). Landmark indices are unchanged (canonical face mesh).
    """
    import mediapipe as mp
    from mediapipe.tasks.python import vision, BaseOptions

    opts = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(ROOT / "models/face_landmarker.task")),
        running_mode=vision.RunningMode.VIDEO, num_faces=1,
    )
    mesh = vision.FaceLandmarker.create_from_options(opts)

    cap = cv2.VideoCapture(str(video_path))
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    fps = orig_fps / stride
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  video: {total:,} frames @ {orig_fps:.2f} fps ({total/orig_fps/60:.1f} min)")

    rgb, geom, ok, i = [], [], 0, 0
    while True:
        ret, frame = cap.read()
        if not ret or (max_frames and i >= max_frames):
            break
        if i % stride:
            i += 1
            continue
        h, w = frame.shape[:2]
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                          data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        res = mesh.detect_for_video(mp_img, int(i * 1000 / orig_fps))
        if res.face_landmarks:
            lm = res.face_landmarks[0]
            y0, y1 = int(lm[LM_TOP].y * h), int(lm[LM_BOTTOM].y * h)
            x0, x1 = int(lm[LM_LEFT].x * w), int(lm[LM_RIGHT].x * w)
            y0, y1 = sorted((max(0, y0), min(h, y1)))
            x0, x1 = sorted((max(0, x0), min(w, x1)))
            pad = int((y1 - y0) * 0.15)                 # stay off hairline / eyebrows
            y0, y1 = y0 + pad, y1 - pad
            roi = frame[y0:y1, x0:x1]
            if roi.size:
                rgb.append(roi.reshape(-1, 3).mean(0)[::-1])          # BGR->RGB
                geom.append([(x0 + x1) / 2 / w, (y0 + y1) / 2 / h,    # centroid (norm)
                             (x1 - x0) * (y1 - y0) / (w * h)])         # ROI area (norm)
                ok += 1
                i += 1
                if i % 5000 == 0:
                    print(f"    {i:,} frames, {ok:,} with face")
                continue
        rgb.append([np.nan] * 3)
        geom.append([np.nan] * 3)
        i += 1
    cap.release()
    mesh.close()
    return np.asarray(rgb, float), np.asarray(geom, float), fps, ok


def _interp_nan(x):
    idx = np.arange(len(x))
    good = np.isfinite(x)
    return np.interp(idx, idx[good], x[good]) if good.sum() >= 10 else None


def pos_pulse(rgb, fps):
    """POS (Wang et al. 2017): motion-robust rPPG from RGB via overlap-add projection."""
    C = np.vstack([_interp_nan(rgb[:, k]) for k in range(3)]).T   # N x 3
    N = len(C)
    l = max(8, int(fps * 1.6))                                    # ~1.6 s window
    H = np.zeros(N)
    for n in range(0, N - l):
        Cn = C[n:n + l] / (C[n:n + l].mean(0) + 1e-8)
        S1 = Cn[:, 1] - Cn[:, 2]                                  # G - B
        S2 = -2 * Cn[:, 0] + Cn[:, 1] + Cn[:, 2]                  # -2R + G + B
        h = S1 + (S1.std() / (S2.std() + 1e-8)) * S2
        H[n:n + l] += h - h.mean()
    return H


def compute_features(rgb, geom, fps):
    g = _interp_nan(rgb[:, 1])
    if g is None:
        raise RuntimeError("too few frames with a detected face")
    gn = signal.detrend(g / (np.nanmean(g) + 1e-8))              # normalized green
    nyq = fps / 2.0

    b_lo, a_lo = signal.butter(2, min(0.05 / nyq, 0.99), btype="low")
    tonic = signal.filtfilt(b_lo, a_lo, gn)

    pulse = pos_pulse(rgb, fps)
    b_bp, a_bp = signal.butter(3, [0.7 / nyq, min(2.5 / nyq, 0.99)], btype="band")
    pulse = signal.filtfilt(b_bp, a_bp, pulse)
    env = np.abs(signal.hilbert(pulse))

    # motion proxy: frame-to-frame change in ROI centroid + area
    gm = np.vstack([_interp_nan(geom[:, k]) for k in range(3)]).T
    motion = np.r_[0, np.abs(np.diff(gm, axis=0)).sum(1)]

    win, step = int(WIN_S * fps), int(STEP_S * fps)
    t, bv, pa, pr, mo = [], [], [], [], []
    for s in range(0, len(gn) - win, step):
        e = s + win
        t.append((s + win / 2) / fps)
        bv.append(tonic[s:e].max() - tonic[s:e].min())
        pa.append(env[s:e].mean())
        mo.append(motion[s:e].mean())
        pk, _ = signal.find_peaks(pulse[s:e], distance=max(1, int(fps * 0.4)))
        pr.append(-60.0 / (np.diff(pk) / fps).mean() if len(pk) > 2 else np.nan)
    return (np.array(t),
            {"blood_volume": np.array(bv), "pulse_amp": np.array(pa), "pulse_rate": np.array(pr)},
            np.array(mo))


def _win_mean(t_stream, v_stream, centres):
    out = []
    for tc in centres:
        m = (t_stream >= tc - WIN_S / 2) & (t_stream < tc + WIN_S / 2)
        out.append(v_stream[m].mean() if m.any() else np.nan)
    return np.array(out)


def load_gsr(subject, centres):
    bio = pd.read_csv(DATA / f"probe_bio/biosignals_raw/{subject}.csv", sep="\t",
                      usecols=["time", "gsr"])
    return _win_mean(bio["time"].to_numpy() / 1e6, bio["gsr"].to_numpy(float), centres)


def load_temperature(subject, centres):
    try:
        z = zipfile.ZipFile(DATA / "partC/temperature.zip")
        n = [x for x in z.namelist() if f"{subject}.csv" in x][0]
        tp = pd.read_csv(io.BytesIO(z.read(n)), sep="\t")
        return _win_mean(tp["time"].to_numpy() / 1e6, tp["temperature"].to_numpy(float), centres)
    except Exception:
        return np.full(len(centres), np.nan)


def partial_spearman(x, y, controls):
    """Spearman partial correlation of x,y controlling for columns of `controls`.

    Ranks are standardized and residualized via the SVD pseudo-inverse, so collinear
    controls (e.g. temperature that drifts monotonically with time) degrade gracefully
    instead of blowing up the least-squares coefficients.
    """
    m = np.isfinite(x) & np.isfinite(y)
    for c in controls:
        m &= np.isfinite(c)
    if m.sum() < 8:
        return np.nan, int(m.sum())

    def zrank(a):
        r = rankdata(a[m]).astype(float)
        r -= r.mean()
        s = r.std()
        return r / s if s > 1e-9 else r

    rx, ry = zrank(x), zrank(y)
    if rx.std() < 1e-9 or ry.std() < 1e-9:
        return np.nan, int(m.sum())
    if not controls:
        return float(np.corrcoef(rx, ry)[0, 1]), int(m.sum())

    Z = np.column_stack([zrank(c) for c in controls])
    Z = Z[:, Z.std(0) > 1e-9]                       # drop degenerate controls
    Z = np.column_stack([Z, np.ones(m.sum())])
    # rcond=1e-6 truncates collinear directions (e.g. temp~time) instead of
    # inverting a ~0 singular value to ~1e15 and overflowing.
    Zp = np.linalg.pinv(Z, rcond=1e-6)
    rxr = rx - Z @ (Zp @ rx)
    ryr = ry - Z @ (Zp @ ry)
    if rxr.std() < 1e-9 or ryr.std() < 1e-9:
        return np.nan, int(m.sum())
    return float(np.corrcoef(rxr, ryr)[0, 1]), int(m.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("subject")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--refresh", action="store_true", help="re-extract even if cached")
    a = ap.parse_args()

    vid = DATA / f"video/{a.subject}.mp4"                        # FIXED path
    if not vid.exists():
        raise SystemExit(f"missing video: {vid}\nPull it with src/fetch_resumable.py {a.subject}")

    print(f"subject: {a.subject}")
    cache = ROOT / "notes" / f"trace_{a.subject}_s{a.stride}.npz"
    if cache.exists() and not a.refresh:
        d = np.load(cache)
        rgb, geom, fps, ok = d["rgb"], d["geom"], float(d["fps"]), int(d["ok"])
        print(f"  loaded cached forehead trace ({len(rgb):,} frames @ {fps:.1f} fps)")
    else:
        rgb, geom, fps, ok = extract_forehead(vid, a.max_frames, a.stride)
        np.savez_compressed(cache, rgb=rgb, geom=geom, fps=fps, ok=ok)
    print(f"  face detected in {ok:,}/{len(rgb):,} frames ({100*ok/max(1,len(rgb)):.1f}%)")

    t, feats, motion = compute_features(rgb, geom, fps)
    gsr = load_gsr(a.subject, t)
    temp = load_temperature(a.subject, t)
    time_trend = t.copy()
    n_indep = int(WIN_S / STEP_S)                                # stride to de-overlap
    print(f"  {len(t)} windows of {WIN_S:.0f}s (~{len(t)//n_indep} independent)\n")

    print(f"{'feature':>14} {'raw r':>8} {'partial r':>10} {'p(indep)':>9}   verdict")
    print(f"{'':>14} {'':>8} {'(motion,temp,time ctrl)':>22}")
    print("-" * 72)
    best_partial = 0.0
    for name, v in feats.items():
        raw_r, _ = partial_spearman(v, gsr, [])
        par_r, nused = partial_spearman(v, gsr, [motion, temp, time_trend])
        # honest significance: non-overlapping windows only
        sub = slice(None, None, n_indep)
        m = np.isfinite(v[sub]) & np.isfinite(gsr[sub])
        p = spearmanr(v[sub][m], gsr[sub][m])[1] if m.sum() >= 5 else np.nan
        if np.isfinite(par_r):
            best_partial = max(best_partial, abs(par_r))
        flag = ("***" if p < 0.01 else "*" if p < 0.05 else "ns") if np.isfinite(p) else "n/a"
        print(f"{name:>14} {raw_r:>+8.3f} {par_r:>+10.3f} {p:>9.4f}   {flag}")

    print(f"\n  sanity — motion vs GSR (should be modest): "
          f"r={partial_spearman(motion, gsr, [])[0]:+.3f}")
    print(f"  ETH reference (100 Hz, chin rest): partial r ~ 0.57-0.63")
    print(f"  best PARTIAL here (25 Hz, free movement): {best_partial:.3f}\n")
    if best_partial > 0.3:
        print("=> SIGNAL SURVIVES CONFOUND CONTROLS. Direction is live; scale to more subjects.")
    else:
        print("=> No confound-robust signal on this subject. Confirm on responders + a")
        print("   flat-GSR negative control before concluding; a real null pivots to rPPG.")


if __name__ == "__main__":
    main()
