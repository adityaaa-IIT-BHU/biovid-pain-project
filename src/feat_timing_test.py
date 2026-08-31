"""Timing probe: how long does py-feat AU detection take per frame on this machine?
Workaround: torchcodec's video backend can't find its ffmpeg dylib on this box, so we
decode frames with cv2 (already used elsewhere in this project) and feed py-feat file paths."""
import tempfile
import time
from pathlib import Path

import cv2
from feat import Detectorv1

print("loading Detector...")
t0 = time.time()
det = Detectorv1(device="cpu")
print(f"  loaded in {time.time()-t0:.1f}s")

cap = cv2.VideoCapture("data/video/071709_w_23.mp4")
tmp = Path(tempfile.mkdtemp())
paths = []
for i in range(10):
    ret, frame = cap.read()
    if not ret:
        break
    p = tmp / f"f{i}.jpg"
    cv2.imwrite(str(p), frame)
    paths.append(str(p))
cap.release()
print(f"wrote {len(paths)} test frames to {tmp}")

t0 = time.time()
out = det.detect(paths, data_type="image", progress_bar=False)
dt = time.time() - t0
print(f"\n{len(paths)} frames in {dt:.1f}s -> {dt/len(paths):.2f}s/frame")
print(f"AU columns: {[c for c in out.columns if 'AU' in c]}")
print(out.filter(like="AU").head())
