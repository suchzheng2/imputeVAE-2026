#!/usr/bin/env python3
"""Phase 1: audit and download external validation datasets from GEO.

Run on the LOGIN node (compute nodes may not reach GEO).
Atomic downloads: temp file -> gzip -t -> sha256 -> move.
Writes outputs/data_manifest.tsv. Never overwrites a valid existing file.

Usage (from project root):
    python validation/trap_ribotag_external_validation/scripts/00_audit_and_download.py
"""
import gzip
import hashlib
import os
import subprocess
import sys
import tarfile
import time
import urllib.request
from datetime import date

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(TASK_DIR, "data")
MANIFEST = os.path.join(TASK_DIR, "outputs", "data_manifest.tsv")
GEO = "https://ftp.ncbi.nlm.nih.gov/geo"

FILES = [
    # (dataset, filename, url)
    ("GSE210071", "GSE210071_FPKM_with_annotations.txt.gz",
     f"{GEO}/series/GSE210nnn/GSE210071/suppl/GSE210071_FPKM_with_annotations.txt.gz"),
    ("GSE210071", "GSE210071-GPL17021_series_matrix.txt.gz",
     f"{GEO}/series/GSE210nnn/GSE210071/matrix/GSE210071-GPL17021_series_matrix.txt.gz"),
    ("GSE210071", "GSE210071-GPL24247_series_matrix.txt.gz",
     f"{GEO}/series/GSE210nnn/GSE210071/matrix/GSE210071-GPL24247_series_matrix.txt.gz"),
    ("GSE121199", "GSE121199_all_counts.txt.gz",
     f"{GEO}/series/GSE121nnn/GSE121199/suppl/GSE121199_all_counts.txt.gz"),
    ("GSE121199", "GSE121199_series_matrix.txt.gz",
     f"{GEO}/series/GSE121nnn/GSE121199/matrix/GSE121199_series_matrix.txt.gz"),
    ("GSE263082", "GSE263082_NS046_raw-counts_gt40.txt.gz",
     f"{GEO}/series/GSE263nnn/GSE263082/suppl/GSE263082_NS046_raw-counts_gt40.txt.gz"),
    ("GSE263082", "GSE263082_Max_NS68_raw.txt.gz",
     f"{GEO}/series/GSE263nnn/GSE263082/suppl/GSE263082_Max_NS68_raw.txt.gz"),
    ("GSE263082", "GSE263082_Batch3_RawGeneCounts.txt.gz",
     f"{GEO}/series/GSE263nnn/GSE263082/suppl/GSE263082_Batch3_RawGeneCounts.txt.gz"),
    ("GSE263082", "GSE263082_Batch4_RawGeneCounts.txt.gz",
     f"{GEO}/series/GSE263nnn/GSE263082/suppl/GSE263082_Batch4_RawGeneCounts.txt.gz"),
    ("GSE263082", "GSE263082_Batch5_RawGeneCounts.txt.gz",
     f"{GEO}/series/GSE263nnn/GSE263082/suppl/GSE263082_Batch5_RawGeneCounts.txt.gz"),
    ("GSE263082", "GSE263082_series_matrix.txt.gz",
     f"{GEO}/series/GSE263nnn/GSE263082/matrix/GSE263082_series_matrix.txt.gz"),
    ("GSE174303", "GSE174303_counts_matrix.txt.gz",
     f"{GEO}/series/GSE174nnn/GSE174303/suppl/GSE174303_counts_matrix.txt.gz"),
    ("GSE174303", "GSE174303_series_matrix.txt.gz",
     f"{GEO}/series/GSE174nnn/GSE174303/matrix/GSE174303_series_matrix.txt.gz"),
    ("GSE13379", "GSE13379_series_matrix.txt.gz",
     f"{GEO}/series/GSE13nnn/GSE13379/matrix/GSE13379_series_matrix.txt.gz"),
    ("GSE13379", "GPL1261.annot.gz",
     f"{GEO}/platforms/GPL1nnn/GPL1261/annot/GPL1261.annot.gz"),
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def gzip_ok(path):
    if not path.endswith(".gz"):
        return True
    r = subprocess.run(["gzip", "-t", path], capture_output=True)
    return r.returncode == 0


def download(url, dest, retries=3):
    tmp = dest + ".part"
    for attempt in range(1, retries + 1):
        try:
            urllib.request.urlretrieve(url, tmp)
            if gzip_ok(tmp) and os.path.getsize(tmp) > 0:
                os.replace(tmp, dest)
                return True
            print(f"    attempt {attempt}: corrupt/empty download", file=sys.stderr)
        except Exception as e:
            print(f"    attempt {attempt}: {e}", file=sys.stderr)
        if os.path.exists(tmp):
            os.remove(tmp)
        time.sleep(5 * attempt)
    return False


def main():
    os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
    rows = []
    today = date.today().isoformat()
    n_fail = 0
    for dataset, fname, url in FILES:
        ddir = os.path.join(DATA_DIR, dataset)
        os.makedirs(ddir, exist_ok=True)
        dest = os.path.join(ddir, fname)
        status = "pre-existing"
        if os.path.exists(dest) and os.path.getsize(dest) > 0 and gzip_ok(dest):
            print(f"[keep]     {dataset}/{fname}")
        else:
            if os.path.exists(dest):
                print(f"[corrupt]  {dataset}/{fname} -> re-downloading")
                os.remove(dest)
            print(f"[download] {dataset}/{fname}")
            if download(url, dest):
                status = f"downloaded {today}"
            else:
                status = "FAILED"
                n_fail += 1
                rows.append((dataset, fname, url, "", "", status, "gzip_ok=NA"))
                continue
        size = os.path.getsize(dest)
        rows.append((dataset, fname, url, str(size), sha256(dest), status,
                     f"gzip_ok={gzip_ok(dest)}"))
    with open(MANIFEST, "w") as f:
        f.write("dataset\tfilename\tsource_url\tsize_bytes\tsha256\tstatus\tintegrity\n")
        for r in rows:
            f.write("\t".join(r) + "\n")
    print(f"\nManifest: {MANIFEST}  ({len(rows)} files, {n_fail} failed)")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
