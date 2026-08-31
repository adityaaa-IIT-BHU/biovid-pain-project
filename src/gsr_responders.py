"""Cross-subject GSR responder analysis over all 87 Part C subjects.

check_gsr_response.py showed one subject has a clean monotonic GSR dose-response to pain
intensity (r=+0.967). Before building any video model we need to know: for how many of the
87 subjects is the electrodermal signal actually pain-linked? The recoverable-from-video
target only exists for responders; non-responders bound the usable cohort (ETH found 4/21
had no significant EDA response).

Method (identical to the validated single-subject probe):
  - stimulus-locked epochs, PRE=5s / POST=15s at 512 Hz, baseline-corrected to pre-onset mean
  - per intensity 1-4: peak of the post-onset mean epoch (peak ΔGSR)
  - r = corr(intensity, peak ΔGSR); responder if r>0.5 AND peak@4 >= AMP_MIN µS

Biosignals are read straight from biosignals_raw.zip (no ~4 GB extraction).
Results are written to notes/gsr_responders.csv so they persist.
"""
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/adityaacharyaresearch/biovid-pain-project/data")
BIO_ZIP = ROOT / "partC/biosignals_raw.zip"
STIM_DIR = ROOT / "partC/stim_x/stimulus"
OUT = Path("/Users/adityaacharyaresearch/biovid-pain-project/notes/gsr_responders.csv")

FS = 512                 # Hz
PRE, POST = 5.0, 15.0    # s around onset
AMP_MIN = 0.10           # µS; minimal peak@intensity-4 to count as a real response
R_MIN = 0.5              # correlation threshold


def analyze(subject, gsr, t_us, stim):
    n_pre, n_post = int(PRE * FS), int(POST * FS)
    epochs = {lab: [] for lab in (1, 2, 3, 4)}
    for _, row in stim.iterrows():
        lab = int(row["label"])
        if lab not in epochs:
            continue
        idx = int(np.searchsorted(t_us, row["time"]))
        if idx - n_pre < 0 or idx + n_post >= len(gsr):
            continue
        seg = gsr[idx - n_pre: idx + n_post].astype(float)
        seg = seg - seg[:n_pre].mean()
        epochs[lab].append(seg)

    peaks, ns = {}, {}
    for lab, segs in epochs.items():
        if not segs:
            continue
        resp = np.mean(segs, axis=0)[n_pre:]
        peaks[lab] = float(resp.max())
        ns[lab] = len(segs)
    if len(peaks) < 2:
        return None
    labs = np.array(sorted(peaks))
    vals = np.array([peaks[l] for l in labs])
    r = float(np.corrcoef(labs, vals)[0, 1])
    peak4 = peaks.get(4, np.nan)
    responder = (r > R_MIN) and (peak4 >= AMP_MIN)
    return dict(subject=subject, r=r, peak1=peaks.get(1, np.nan), peak2=peaks.get(2, np.nan),
                peak3=peaks.get(3, np.nan), peak4=peak4, n_stim=sum(ns.values()),
                responder=responder)


def main():
    z = zipfile.ZipFile(BIO_ZIP)
    bio_files = {Path(n).stem: n for n in z.namelist() if n.endswith(".csv")}
    rows, skipped = [], []
    for i, subject in enumerate(sorted(bio_files), 1):
        stim_path = STIM_DIR / f"{subject}.csv"
        if not stim_path.exists():
            skipped.append((subject, "no stimulus file"))
            continue
        try:
            bio = pd.read_csv(io.BytesIO(z.read(bio_files[subject])), sep="\t",
                              usecols=["time", "gsr"])
            stim = pd.read_csv(stim_path, sep="\t")
            res = analyze(subject, bio["gsr"].to_numpy(), bio["time"].to_numpy(), stim)
            if res is None:
                skipped.append((subject, "too few epochs"))
                continue
            rows.append(res)
            flag = "RESP " if res["responder"] else "  -  "
            print(f"  [{i:2}/87] {subject:<14} r={res['r']:+.3f}  peak@4={res['peak4']:+.3f}µS  {flag}")
        except Exception as e:
            skipped.append((subject, str(e)[:60]))
            print(f"  [{i:2}/87] {subject:<14} ERROR: {str(e)[:50]}")

    df = pd.DataFrame(rows).sort_values("r", ascending=False)
    OUT.parent.mkdir(exist_ok=True)
    df.to_csv(OUT, index=False)

    n = len(df)
    nr = int(df["responder"].sum())
    print("\n" + "=" * 60)
    print(f"SUMMARY  ({n} subjects analyzed, {len(skipped)} skipped)")
    print("=" * 60)
    print(f"  responders (r>{R_MIN} & peak@4>={AMP_MIN}µS): {nr}/{n}  ({100*nr/n:.0f}%)")
    print(f"  non-responders                             : {n-nr}/{n}")
    print(f"  median r across all subjects               : {df['r'].median():+.3f}")
    print(f"  median r among responders                  : {df[df.responder]['r'].median():+.3f}")
    print(f"\n  saved -> {OUT}")
    if skipped:
        print(f"  skipped: {skipped}")


if __name__ == "__main__":
    main()
