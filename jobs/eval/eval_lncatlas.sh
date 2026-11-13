#!/bin/bash
#SBATCH --job-name=lncatlas
#SBATCH --output=jobs/logs/slurm-%j_lncatlas.out
#SBATCH --error=jobs/logs/slurm-%j_lncatlas.err
#SBATCH --partition=scavenge
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00

echo "Job started: $(date)"
echo "Node: $(hostname)"

module load miniconda
conda activate impute2

cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE

find imputeVAE/ analysis/ -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null

LNCATLAS_CSV="data/external/lncATLAS_all_data.csv"

# Step 0: Download lncATLAS if not present or too small
if [ ! -f "$LNCATLAS_CSV" ] || [ $(stat -c%s "$LNCATLAS_CSV" 2>/dev/null || echo 0) -lt 1000 ]; then
    echo "=== Downloading lncATLAS data ==="
    # lncATLAS is a Shiny app — no direct CSV URL.
    # Try fetching via the Shiny download handler.
    python3 - << 'PYEOF'
import os, sys, time, json, requests
from websocket import create_connection

OUT = "data/external/lncATLAS_all_data.csv"

print("Connecting to lncATLAS Shiny app...")
# 1. Get initial page to find the websocket URL
s = requests.Session()
r = s.get("https://lncatlas.crg.eu/", timeout=30)
assert r.status_code == 200, f"Failed to load page: {r.status_code}"

# Shiny apps use a websocket at the same host
# The initial connection negotiates a session ID
import re

# Try to get a Shiny token
# Shiny 0.13 uses a reconnect token in a meta tag or via __shiny__
# We need to open a websocket session to trigger the download

print("  Page loaded. Attempting websocket session...")

# Shiny websocket endpoint
ws_url = "wss://lncatlas.crg.eu/websocket/"

try:
    ws = create_connection(ws_url, timeout=30,
                          header={"Origin": "https://lncatlas.crg.eu"})
    print(f"  WebSocket connected")

    # Read initial message
    init = ws.recv()
    print(f"  Init message length: {len(init)}")

    # Send a message requesting "download all data"
    # Shiny download buttons have output IDs; we need to find the right one
    # Typically it's something like "downloadAllData" or "rawDataDownload"

    ws.close()
    print("  WebSocket approach needs interactive session. Trying alternative...")
except Exception as e:
    print(f"  WebSocket failed: {e}")
    print("  This is expected — lncATLAS requires an interactive browser session.")

# Alternative: Build CN-RCI from ENCODE data via the supplementary tables
# Actually, let's try the simplest thing: use the REST-like Shiny URL pattern
print("\nTrying alternative download approaches...")
for url in [
    "https://lncatlas.crg.eu/data/lncATLAS_all_data.csv",
    "https://lncatlas.crg.eu/lncATLAS-app/data/lncATLAS_all_data.csv",
    "https://lncatlas.crg.eu/session/download?w=&output=downloadAllData",
]:
    try:
        r = s.get(url, timeout=15, stream=True)
        if r.status_code == 200 and len(r.content) > 1000:
            with open(OUT, "wb") as f:
                f.write(r.content)
            print(f"  SUCCESS: Downloaded {len(r.content)} bytes from {url}")
            sys.exit(0)
        else:
            print(f"  {r.status_code} from {url} ({len(r.content)} bytes)")
    except Exception as e:
        print(f"  Failed {url}: {e}")

print("\nAutomatic download failed. lncATLAS requires browser interaction.")
print("Please download manually:")
print("  1. Go to https://lncatlas.crg.eu/")
print("  2. Click 'Get Raw Data' tab")
print("  3. Click 'Download All raw data'")
print(f"  4. Save to {os.path.abspath(OUT)}")
sys.exit(1)
PYEOF

    DOWNLOAD_STATUS=$?
    if [ $DOWNLOAD_STATUS -ne 0 ]; then
        echo ""
        echo "=== lncATLAS auto-download failed ==="
        echo "The lncATLAS website requires browser interaction to download."
        echo "Please download manually and re-run this job."
        echo ""
        echo "FALLBACK: Running ortholog mapping + caching only (useful for next run)..."
        python3 -c "
import sys; sys.path.insert(0, '.')
from analysis.localization_lncatlas import get_mouse_human_orthologs
orth = get_mouse_human_orthologs()
if orth is not None:
    print(f'Ortholog cache ready: {len(orth)} mappings')
"
        echo ""
        echo "After downloading lncATLAS_all_data.csv, re-submit this job."
        exit 1
    fi
fi

echo ""
echo "=== Running lncATLAS concordance analysis ==="
python analysis/localization_lncatlas.py \
    --lncatlas_csv "$LNCATLAS_CSV"

echo ""
echo "Job finished: $(date)"
