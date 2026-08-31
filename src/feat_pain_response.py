"""Does the FACE visibly react to pain intensity? The PSPI check.

Facial EMG (corrugator/zygomaticus) turned out to be empty in this dataset (0/87 subjects
have any signal in those columns) -- confirmed by direct inspection, not a bug. So we can't
validate facial-expression pain response against EMG ground truth. But the expression itself
is VISIBLE in the video, so we score it directly with py-feat's Action Unit detector and the
published pain-intensity formula (Prkachin & Solomon 2008):

    PSPI = AU04 + max(AU06, AU07) + max(AU09, AU10) + AU43

  AU04 = brow lowerer      (the "corrugator" muscle -- exactly what the missing EMG channel
                             would have measured, now read directly from video instead)
  AU06/07 = cheek raiser / lid tightener
  AU09/10 = nose wrinkler / upper lip raiser
  AU43    = eye closure

Same stimulus-locked dose-response design as the GSR/HR/EMG checks: baseline-correct each
pain-onset epoch, take the post-onset peak, correlate peak vs. intensity across 4 levels.

py-feat is slow on CPU (~1.5s/frame), so this samples SPARSELY within each epoch (every 2s,
not every video frame) on ONE subject as a time-boxed first check, not the full 87-subject
power of the other analyses.
"""
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from feat import Detectorv1

ROOT = Path("/Users/adityaacharyaresearch/biovid-pain-project")
SUBJECT = "071709_w_23"
FPS = 25.0
PRE, POST = 4.0, 14.0          # seconds around onset (slightly tighter than the bio version)
STEP = 2.0                     # sample spacing within the epoch -- the compute/resolution tradeoff


def sample_offsets():
    return np.arange(-PRE, POST + 1e-6, STEP)


def main():
    stim = pd.read_csv(ROOT / f"data/partC/stim_x/stimulus/{SUBJECT}.csv", sep="\t")
    cap = cv2.VideoCapture(str(ROOT / f"data/video/{SUBJECT}.mp4"))
    offsets = sample_offsets()
    print(f"subject {SUBJECT}: {len(stim)} stimuli, {len(offsets)} samples/epoch "
          f"({offsets.min():+.0f}s to {offsets.max():+.0f}s, every {STEP:.0f}s)")

    tmp = Path(tempfile.mkdtemp())
    frame_paths, meta = [], []          # meta: (stim_idx, intensity, offset)
    for si, row in stim.iterrows():
        t0 = row["time"] / 1e6
        for off in offsets:
            fidx = int(round((t0 + off) * FPS))
            cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
            ret, frame = cap.read()
            if not ret:
                continue
            p = tmp / f"s{si}_o{off:+.1f}.jpg"
            cv2.imwrite(str(p), frame)
            frame_paths.append(str(p))
            meta.append((si, int(row["label"]), off))
    cap.release()
    print(f"extracted {len(frame_paths)} frames -> running AU detection "
          f"(~{len(frame_paths)*1.5/60:.0f} min at ~1.5s/frame)...")

    det = Detectorv1(device="cpu")
    t0 = time.time()
    out = det.detect(frame_paths, data_type="image", progress_bar=False)
    print(f"  AU detection done in {(time.time()-t0)/60:.1f} min")

    out = out.reset_index(drop=True)
    out["stim_idx"] = [m[0] for m in meta]
    out["intensity"] = [m[1] for m in meta]
    out["offset"] = [m[2] for m in meta]
    out["PSPI"] = (out["AU04"] + out[["AU06", "AU07"]].max(axis=1)
                  + out[["AU09", "AU10"]].max(axis=1) + out["AU43"])
    out.to_csv(ROOT / "notes/feat_au_raw.csv", index=False)

    # per-epoch: baseline-correct PSPI (and AU04 alone) to the pre-onset mean, take post-onset peak
    rows = []
    for si, g in out.groupby("stim_idx"):
        pre = g[g.offset < 0]
        post = g[g.offset >= 0]
        if len(pre) < 1 or len(post) < 1:
            continue
        base_pspi, base_au4 = pre["PSPI"].mean(), pre["AU04"].mean()
        rows.append(dict(stim_idx=si, intensity=g["intensity"].iloc[0],
                         peak_pspi=post["PSPI"].max() - base_pspi,
                         peak_au4=post["AU04"].max() - base_au4))
    ep = pd.DataFrame(rows)
    ep.to_csv(ROOT / "notes/feat_pain_epochs.csv", index=False)

    print("\n" + "=" * 60)
    print(f"FACIAL EXPRESSION PAIN DOSE-RESPONSE  (subject {SUBJECT}, {len(ep)} epochs)")
    print("=" * 60)
    for metric, label in [("peak_pspi", "PSPI (full pain formula)"),
                          ("peak_au4", "AU04 alone (brow lowerer / corrugator-proxy)")]:
        by_int = ep.groupby("intensity")[metric].mean()
        r = np.corrcoef(by_int.index, by_int.values)[0, 1]
        print(f"\n  {label}:")
        for i, v in by_int.items():
            print(f"    intensity {i}: mean peak Δ = {v:+.3f}  (n={int((ep.intensity==i).sum())})")
        print(f"    corr(intensity, peak) = {r:+.3f}")

    print(f"\n  saved -> notes/feat_au_raw.csv, notes/feat_pain_epochs.csv")


if __name__ == "__main__":
    main()
