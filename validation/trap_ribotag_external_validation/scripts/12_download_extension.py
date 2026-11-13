#!/usr/bin/env python3
"""Extension (2026-09-18): download newer RNA-seq RiboTag/TRAP datasets. Login node only.
Atomic download -> integrity check (gzip -t / tar -tf) -> sha256 -> outputs/extension/data_manifest_extension.tsv.
GSE133291 (Furlanis 2019) is reused from <project>/data/external/furlanis2019 and only checksummed."""
import hashlib, os, subprocess, sys, tarfile, time, urllib.request
from datetime import date
TASK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT = os.path.dirname(os.path.dirname(TASK))
DATA = f"{TASK}/data"; MAN = f"{TASK}/outputs/extension/data_manifest_extension.tsv"
G = "https://ftp.ncbi.nlm.nih.gov/geo/series"
def s(acc): return f"{G}/{acc[:-3]}nnn/{acc}"
FILES = []
for acc in ["GSE172484", "GSE202172", "GSE141337", "GSE142028", "GSE141874", "GSE139684"]:
    FILES.append((acc, f"{acc}_RAW.tar", f"{s(acc)}/suppl/{acc}_RAW.tar"))
FILES += [("GSE99791", "GSE99791_raw.txt.gz", f"{s('GSE99791')}/suppl/GSE99791_raw.txt.gz"),
          ("GSE211162", "GSE211162_Oligodendrocyte_TRAP-seq_RPKM.csv.gz", f"{s('GSE211162')}/suppl/GSE211162_Oligodendrocyte_TRAP-seq_RPKM.csv.gz")]
for acc in ["GSE133291", "GSE99791", "GSE139684"]:
    FILES.append((acc, f"{acc}_series_matrix.txt.gz", f"{s(acc)}/matrix/{acc}_series_matrix.txt.gz"))
for acc, plats in [("GSE211162", ["GPL17021", "GPL21103"]), ("GSE172484", ["GPL17021", "GPL19057"])]:
    for p in plats:
        FILES.append((acc, f"{acc}-{p}_series_matrix.txt.gz", f"{s(acc)}/matrix/{acc}-{p}_series_matrix.txt.gz"))
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""): h.update(c)
    return h.hexdigest()
def ok(p):
    if p.endswith(".gz"): return subprocess.run(["gzip", "-t", p]).returncode == 0
    if p.endswith(".tar"): return tarfile.is_tarfile(p)
    return os.path.getsize(p) > 0
rows = []; fail = 0
for acc, fn, url in FILES:
    d = f"{DATA}/{acc}"; os.makedirs(d, exist_ok=True); dest = f"{d}/{fn}"; status = "pre-existing"
    if not (os.path.exists(dest) and os.path.getsize(dest) > 0 and ok(dest)):
        tmp = dest + ".part"; got = False
        for k in range(3):
            try:
                urllib.request.urlretrieve(url, tmp)
                if os.path.getsize(tmp) > 0 and ok(tmp): os.replace(tmp, dest); got = True; break
            except Exception as e: print(f"  retry {k+1} {fn}: {e}", file=sys.stderr)
            time.sleep(5)
        status = f"downloaded {date.today()}" if got else "FAILED"
        if not got: fail += 1; rows.append((acc, fn, url, "", "", status)); print("[FAIL]", fn); continue
    rows.append((acc, fn, url, str(os.path.getsize(dest)), sha(dest), status)); print(f"[{status}] {acc}/{fn}")
    if fn.endswith("_RAW.tar"):
        out = f"{d}/raw"; os.makedirs(out, exist_ok=True)
        with tarfile.open(dest) as t: t.extractall(out)
fur = f"{PROJECT}/data/external/furlanis2019/GSE133291_selectedCounts.txt.gz"
rows.append(("GSE133291", os.path.basename(fur), f"{s('GSE133291')}/suppl/GSE133291_selectedCounts.txt.gz", str(os.path.getsize(fur)), sha(fur), "pre-existing in data/external/furlanis2019 (downloaded 2026-08-05)"))
with open(MAN, "w") as f:
    f.write("dataset\tfilename\tsource_url\tsize_bytes\tsha256\tstatus\n"); [f.write("\t".join(r) + "\n") for r in rows]
print(f"manifest {MAN}: {len(rows)} files, {fail} failed"); sys.exit(1 if fail else 0)
