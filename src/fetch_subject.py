"""Pull one subject's video + biosignals out of the BioVid Part C archives.

The OVGU Nextcloud supports HTTP range requests, so we extract individual members
from the 76 GB video.zip without downloading it. At ~130 KB/s one subject is ~2 h;
on a fast link it is minutes.

Usage:
    python3 src/fetch_subject.py 071709_w_23
    python3 src/fetch_subject.py --list
"""
import argparse
import os
import sys
import time
from pathlib import Path

import requests
from remotezip import RemoteZip

BASE = "https://cloud.ovgu.de/public.php/webdav/PartC"
TOKEN = os.environ["BIOVID_TOKEN"]
PASSWORD = os.environ["BIOVID_PASSWORD"]  # rotates; email sascha.gruss@uni-ulm.de if it stops working
DATA = Path("/Users/adityaacharyaresearch/biovid-pain-project/data")


def session():
    s = requests.Session()
    s.auth = (TOKEN, PASSWORD)
    return s


def fetch(archive, member, dest):
    dest.mkdir(parents=True, exist_ok=True)
    with RemoteZip(f"{BASE}/{archive}", session=session()) as z:
        hit = [i for i in z.infolist() if member in i.filename and not i.is_dir()]
        if not hit:
            raise SystemExit(f"not found in {archive}: {member}")
        info = hit[0]
        out = dest / Path(info.filename).name
        if out.exists() and out.stat().st_size == info.file_size:
            print(f"  cached: {out.name} ({info.file_size/1e6:.0f} MB)")
            return out
        mb = info.file_size / 1e6
        print(f"  fetching {info.filename} ({mb:.0f} MB) — ~{mb/0.13/60:.0f} min at 130 KB/s")
        t0 = time.time()
        z.extract(info.filename, path=dest.parent)
        dt = time.time() - t0
        print(f"  done in {dt/60:.1f} min ({mb/max(dt,1)*1000:.0f} KB/s)")
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("subject", nargs="?")
    ap.add_argument("--list", action="store_true", help="list subjects by video size")
    ap.add_argument("--video-only", action="store_true")
    a = ap.parse_args()

    if a.list:
        with RemoteZip(f"{BASE}/video.zip", session=session()) as z:
            rows = sorted(
                ((i.filename.split("/")[-1].replace(".mp4", ""), i.file_size)
                 for i in z.infolist() if i.filename.endswith(".mp4")),
                key=lambda r: r[1],
            )
        print(f"{len(rows)} subjects (smallest first — cheapest to probe):\n")
        for n, s in rows:
            print(f"  {s/1e6:7.0f} MB  {n}")
        return

    if not a.subject:
        sys.exit("give a subject name, or --list")

    print(f"subject: {a.subject}")
    if not a.video_only:
        print("biosignals:")
        fetch("biosignals_raw.zip", f"{a.subject}.csv", DATA / "probe_bio")
    print("video:")
    fetch("video.zip", f"{a.subject}.mp4", DATA / "probe_video")
    print("\nnext:")
    print(f"  python3 src/probe_eda_from_video.py {a.subject}")


if __name__ == "__main__":
    main()
