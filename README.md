# BioVid Pain — Camera-Derived Sympathetic Arousal

Can pain-related **electrodermal (sympathetic) arousal** be recovered from face video alone,
where contact GSR isn't available? This project builds and evaluates that pipeline end to end
on the [BioVid Heat Pain Database](https://www.nit.ovgu.de/BioVid.html) Part C, and reports a
clean negative result together with the reasoning and controls behind it.

## Motivation

On BioVid, video is the *weak* modality for automatic pain recognition relative to
physiological sensors. PainFormer (2025), current SOTA on Part A binary classification:

| Modality | Accuracy |
|---|---|
| RGB video | 76.3% |
| ECG | 75.5% |
| **GSR (electrodermal)** | **89.0%** |
| RGB + thermal + depth + GSR | 89.1% |

Adding all video streams on top of GSR buys only +0.09% — video is largely redundant once you
have electrodes. Prior work recovering physiology from video for this task (rSTAN, IJCAI 2021)
targets the *cardiac* signal (rPPG) as an auxiliary task and gains **+0.6%** binary / **+0.4%**
5-class — small, because cardiac signal is the *weaker* pain predictor (ECG ≈ 75%). The channel
that actually carries pain is electrodermal (GSR ≈ 89%), and a separate line of work (ETH SIPLab
peripheral blood-flow rEDA, SympCam, LumEDA) shows sympathetic arousal *can* be read from video
in controlled settings — just never applied to this problem.

This project asks directly: can that electrodermal signal be recovered from BioVid's harder,
freely-moving, 25 Hz frontal-face video regime, and used as a pain-relevant target?

## Approach

Uses **Part C** (one continuous ~25 min video per subject, 87 subjects, with time-aligned
continuous ECG/GSR/EMG/temperature and stimulus-onset labels) rather than Part A's 5.5 s clips,
because electrodermal responses (SCR latency 1–3 s, recovery 2–10 s) need longer windows.

1. **Validate the target signal exists** — stimulus-locked GSR dose-response, confirming pain
   intensity drives skin-conductance response before trying to recover anything from video.
2. **Positive control** — confirm the video pipeline recovers a signal known to be present
   (cardiac pulse, via POS rPPG) before trusting a null result on the harder signal.
3. **EDA-from-video probe** — adapt the ETH SIPLab forehead-blood-flow method to BioVid's
   regime, with explicit motion/lighting/thermode confound controls (partial correlation,
   non-overlapping windows, both-sign significance testing, Stouffer combination across
   subjects) so a positive result can't be an artifact.
4. **Bound the fallback route** — if electrodermal isn't recoverable, quantify how far the
   cardiac (rPPG → HR → pain) route can go instead, and why.

## Key findings

**GSR carries pain across the cohort.** 82/87 Part C subjects (94%) show correct dose-response
ordering (r > 0.5) to stimulus intensity, median r = +0.89. ~70% are strong "responders"
(r > 0.5 and peak response ≥ 0.10 µS) — the rest are electrodermally quiet, bounding the usable
cohort. This confirms the target signal is real and worth recovering.

**The pipeline recovers the cardiac signal from 25 Hz video (positive control).** Video POS-rPPG
pulse rate correlates with ECG heart rate across all probed subjects — the pipeline demonstrably
extracts a real physiological signal from this video regime.

**The pipeline does *not* recover the electrodermal signal (primary result).** Raw
forehead-blood-flow features do correlate with GSR, but that correlation is a **motion
artifact**: after partialling out head motion, the effect collapses to ~zero and flips sign
across subjects (Stouffer-combined z = −0.32, p = 0.63; mean signed partial r = −0.04). Motion
and GSR are themselves correlated (up to r = 0.79) because pain-evoked head motion is confounded
with arousal — so slow forehead-brightness changes track movement, not skin conductance, in this
freely-moving regime.

**Quantitative bound on the cardiac fallback.** Rigorously validated (short sliding windows,
Bland–Altman, not the flattering long-window correlation of a naive check): video-HR vs ECG-HR
agreement is only moderate (MAE ≈ 6 bpm, 95% limits of agreement ≈ ±20 bpm). Meanwhile the
pain-related HR modulation itself is small (peak ΔHR ≈ 3.5 bpm at the highest stimulus
intensity, from the ECG ground truth). Since the pain-related signal (~3.5 bpm) is *smaller*
than the pipeline's own recovery error (~6 bpm), the video→HR→pain route is fundamentally
bounded by noise, not just weak upstream signal. Head-to-head across the full cohort, GSR
(median r 0.89, 70% responders) is a clearly stronger pain channel than HR (median r 0.76, 57%
responders), consistent with the literature.

**Interpretation.** At 25 Hz on freely-moving BioVid faces, the cardiac signal is recoverable
but the electrodermal/sympathetic signal is not — the confound (motion↔arousal) is structural to
this video regime, not a fixable engineering gap. This extends rSTAN's result: they showed
recovering cardiac signal barely helps pain recognition; this project shows *why* the more
promising electrodermal route can't rescue it, and bounds what the cardiac fallback can achieve
instead.

### Caveats

- Forehead-only (no palm, which ETH found to be the best site) — a BioVid limitation, not
  fixable from this side.
- One EDA-feature family tested (blood-flow/luminance); sweat-luminance (LumEDA-style) features
  are untested, though unlikely to escape the same motion confound at this regime.
- Tonic GSR evaluated at coarse window means; phasic-SCR alignment untested.
- Cohort size for the EDA-from-video probe is smaller than the 87-subject cohort used for the
  GSR/HR dose-response and responder analysis; the latter is the full-cohort result.

## Repository structure

```
src/
  check_gsr_response.py      # does GSR track pain intensity at all? (single-subject validation)
  gsr_responders.py          # cross-subject GSR responder analysis, all 87 subjects
  positive_control_hr.py     # positive control: recover cardiac signal (rPPG) from video
  rppg_validation.py         # rigorous rPPG-vs-ECG validation (sliding windows, Bland-Altman)
  probe_eda_from_video.py    # main probe: recover electrodermal signal from forehead video
  probe_summary.py           # multi-subject significance summary with confound controls
  hr_pain_response.py        # ceiling check: does heart rate itself respond to pain intensity?
  pos_hr_pain_response.py    # does POS-recovered video HR track pain intensity directly?
  modality_pain_response.py  # head-to-head: GSR vs HR vs facial EMG as pain channels
  feat_pain_response.py      # facial Action Unit (PSPI) pain response, via py-feat
  feat_timing_test.py        # py-feat detector timing probe
  scale_10subjects.py        # scales the video-based checks from n=1-3 to n=10 subjects
  fetch_subject.py           # pull one subject's video/biosignals from the OVGU archive
  fetch_resumable.py         # stall-proof, resumable range-request fetch for large video members
  test.ipynb, learning.ipynb # exploratory notebooks (outputs stripped before publishing)
models/
  face_landmarker.task       # MediaPipe face landmarker model (not included, see Setup)
```

`data/` (the dataset) and `notes/` (per-subject traces, figures, logs — derived from the
dataset) are intentionally not part of this repository; see **Data access** below.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download the MediaPipe face landmarker model into `models/face_landmarker.task`:
[MediaPipe Face Landmarker docs](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker).

## Data access

This project uses **BioVid Heat Pain Database Part C**, distributed by the University of
Ulm/Magdeburg under a signed data-use agreement — it is **not** included in this repository and
is not redistributable. Request access directly from the dataset maintainers
([nit.ovgu.de/BioVid.html](https://www.nit.ovgu.de/BioVid.html)).

The fetch scripts (`fetch_subject.py`, `fetch_resumable.py`, `scale_10subjects.py`) expect your
own access credentials as environment variables:

```bash
export BIOVID_TOKEN=...
export BIOVID_PASSWORD=...
```

## Key references

- PainFormer (2025) — [arxiv.org/html/2505.01571](https://arxiv.org/html/2505.01571)
- rSTAN, IJCAI 2021 — [ijcai.org/proceedings/2021/0170.pdf](https://www.ijcai.org/proceedings/2021/0170.pdf)
- ETH SIPLab video-based sympathetic arousal — [pmc.ncbi.nlm.nih.gov/articles/PMC10898569](https://pmc.ncbi.nlm.nih.gov/articles/PMC10898569/)
- SympCam (2024) — [arxiv.org/html/2410.20552v1](https://arxiv.org/html/2410.20552v1)
- LumEDA — [iopscience.iop.org/article/10.1088/1361-6579/adb369](https://iopscience.iop.org/article/10.1088/1361-6579/adb369)
- Prkachin & Solomon (2008) — PSPI facial pain-intensity scale
- Werner et al., ICPR 2014 — BioVid preprocessing methodology
