"""Rigorous multi-subject summary of the EDA-from-video probe.

The per-subject probe reports a partial r but tests significance on the RAW correlation and
uses abs() in its verdict — both wrong. This aggregates all cached subjects correctly:

  * partials are computed on NON-OVERLAPPING 60 s windows, so the samples are independent and
    the standard partial-correlation t-test applies: df = n - 2 - k_controls.
  * sign is respected. blood_volume has a directional hypothesis (more forehead blood-volume
    change -> more arousal -> higher GSR), so only a POSITIVE, significant partial supports it.
  * two control sets: motion-only (what you can remove at test time) and motion+temp+time
    (strict). A real effect should survive both.
  * across subjects, blood_volume partials are combined with Stouffer's z.

Primary endpoint: blood_volume under motion-only control.
"""
import glob
import importlib.util
from pathlib import Path

import numpy as np
from scipy.stats import t as tdist, norm

np.seterr(all="ignore")
ROOT = Path("/Users/adityaacharyaresearch/biovid-pain-project")

spec = importlib.util.spec_from_file_location("probe", ROOT / "src/probe_eda_from_video.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)

FEATURES = ["blood_volume", "pulse_amp", "pulse_rate"]
N_INDEP = int(probe.WIN_S / probe.STEP_S)          # stride to de-overlap windows


def partial_p(x, y, controls, n_eff):
    """Partial Spearman r and its t-test p on n_eff independent samples."""
    r, _ = probe.partial_spearman(x, y, controls)
    if not np.isfinite(r):
        return np.nan, np.nan
    df = n_eff - 2 - len(controls)
    if df < 2 or abs(r) >= 1:
        return r, np.nan
    tstat = r * np.sqrt(df / (1 - r * r))
    return r, float(2 * tdist.sf(abs(tstat), df))


def main():
    subjects = sorted(p.split("trace_")[1].rsplit("_s", 1)[0]
                      for p in glob.glob(str(ROOT / "notes/trace_*_s2.npz")))
    print(f"subjects: {subjects}\n")

    rows = {f: [] for f in FEATURES}
    print(f"{'subject':14} {'mot↔GSR':>8}  |  "
          + "  ".join(f"{f[:10]:>10} (|mot  |all)" for f in FEATURES))
    print("-" * 96)
    for s in subjects:
        d = np.load(ROOT / f"notes/trace_{s}_s2.npz")
        t, feats, motion = probe.compute_features(d["rgb"], d["geom"], float(d["fps"]))
        gsr = probe.load_gsr(s, t)
        temp = probe.load_temperature(s, t)
        # independent (non-overlapping) subset for honest inference
        sub = slice(None, None, N_INDEP)
        gI, mI, tempI, tI = gsr[sub], motion[sub], temp[sub], t[sub]
        n_eff = len(gI)
        mgsr, _ = probe.partial_spearman(motion, gsr, [])

        cells = []
        for f in FEATURES:
            vI = feats[f][sub]
            r_m, p_m = partial_p(vI, gI, [mI], n_eff)
            r_a, p_a = partial_p(vI, gI, [mI, tempI, tI], n_eff)
            rows[f].append((s, r_m, p_m, r_a, p_a, n_eff))
            star = "*" if (np.isfinite(p_m) and p_m < 0.05) else " "
            cells.append(f"{r_m:>+5.2f}{star} {r_a:>+5.2f}")
        print(f"{s:14} {mgsr:>+8.2f}  |  " + "  ".join(f"{c:>16}" for c in cells))

    print("\n" + "=" * 60)
    print("PRIMARY ENDPOINT: blood_volume, motion-only control")
    print("=" * 60)
    bv = rows["blood_volume"]
    rs = np.array([r for _, r, _, _, _, _ in bv])
    ps = np.array([p for _, _, p, _, _, _ in bv])
    pos_sig = [(s, r, p) for s, r, p, *_ in bv if np.isfinite(p) and p < 0.05 and r > 0]
    print(f"  per-subject partial r: {np.round(rs, 3).tolist()}")
    print(f"  positive & significant (p<.05): {len(pos_sig)}/{len(bv)}")
    # Stouffer combined z of SIGNED one-sided p (H1: r>0)
    z = []
    for _, r, p, *_ in bv:
        if np.isfinite(p):
            one = (p / 2) if r > 0 else (1 - p / 2)
            z.append(norm.isf(one))
    if z:
        zc = np.sum(z) / np.sqrt(len(z))
        print(f"  Stouffer combined z (H1: video↑ -> GSR↑): {zc:+.2f}  (p={norm.sf(zc):.3f})")
    print(f"\n  mean signed partial r across subjects: {rs[np.isfinite(rs)].mean():+.3f}")
    verdict = ("SUPPORTED" if len(pos_sig) >= 2 else
               "NOT SUPPORTED — no consistent positive motion-robust signal")
    print(f"\n  VERDICT: {verdict}")


if __name__ == "__main__":
    main()
