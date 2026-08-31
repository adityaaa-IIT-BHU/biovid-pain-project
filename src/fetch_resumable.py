"""Resumable, stall-proof fetch of one STORED zip member from OVGU Nextcloud.

The video.zip members are STORED (uncompressed), so the raw mp4 bytes sit contiguously
in the archive. RemoteZip's one-shot extract stalls and cannot resume; this instead:
  1. reads the member's local file header to find the exact byte offset of the mp4 data,
  2. GETs that byte range in chunks with a read timeout,
  3. resumes from whatever is already on disk after any stall/error, with backoff.

Usage:  python3 src/fetch_resumable.py 071709_w_23
"""
import os
import struct
import sys
import time
from pathlib import Path

import requests
from remotezip import RemoteZip

BASE = "https://cloud.ovgu.de/public.php/webdav/PartC"
AUTH = (os.environ["BIOVID_TOKEN"], os.environ["BIOVID_PASSWORD"])  # rotates; email sascha.gruss@uni-ulm.de if it stops working
DATA = Path("/Users/adityaacharyaresearch/biovid-pain-project/data")
CHUNK = 4 * 1024 * 1024        # 4 MB per range request; a stall costs at most this
READ_TIMEOUT = 30             # s; treat a silent socket as a stall and retry


def data_start(sess, url, header_offset):
    """Local file header = 30 fixed bytes + name + extra; data follows."""
    r = sess.get(url, headers={"Range": f"bytes={header_offset}-{header_offset+29}"}, timeout=30)
    r.raise_for_status()
    h = r.content
    name_len, extra_len = struct.unpack("<HH", h[26:30])
    return header_offset + 30 + name_len + extra_len


def main():
    subj = sys.argv[1]
    sess = requests.Session()
    sess.auth = AUTH
    url = f"{BASE}/video.zip"

    with RemoteZip(url, session=sess) as z:
        info = [x for x in z.infolist() if f"{subj}.mp4" in x.filename][0]
        assert info.compress_type == 0, "member is not STORED; need inflate path"
        total = info.file_size
        start = data_start(sess, url, info.header_offset)

    out = DATA / "video" / f"{subj}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    have = out.stat().st_size if out.exists() else 0
    print(f"subject {subj}: {total/1e6:.0f} MB total, resuming from {have/1e6:.0f} MB", flush=True)

    attempt = 0
    while have < total:
        lo = start + have
        hi = min(start + have + CHUNK, start + total) - 1
        try:
            r = sess.get(url, headers={"Range": f"bytes={lo}-{hi}"},
                         stream=True, timeout=(15, READ_TIMEOUT))
            r.raise_for_status()
            with open(out, "ab") as f:
                for block in r.iter_content(256 * 1024):
                    if block:
                        f.write(block)
            have = out.stat().st_size
            attempt = 0
            pct = 100 * have / total
            print(f"  {have/1e6:6.0f}/{total/1e6:.0f} MB ({pct:4.1f}%)", flush=True)
        except Exception as e:
            attempt += 1
            wait = min(30, 2 ** attempt)
            print(f"  stall/err at {have/1e6:.0f} MB ({str(e)[:40]}); retry in {wait}s", flush=True)
            time.sleep(wait)
            have = out.stat().st_size  # keep whatever landed

    print(f"DONE: {out} ({out.stat().st_size/1e6:.0f} MB)", flush=True)
    print(f"next: python3 src/probe_eda_from_video.py {subj}", flush=True)


if __name__ == "__main__":
    main()
