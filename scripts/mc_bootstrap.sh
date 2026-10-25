#!/bin/bash
# mc_bootstrap.sh — unduh, konfigurasi, dan nyalakan server Bedrock.
# Dijalankan SEBAGAI user pemilik server (sudo -u runzzx1 -H), jadi
# semua file di $HOME/mc bisa diurus lewat SSH tanpa sudo.
#
#   MC_DIR   default $HOME/mc
#   MCVER    versi spesifik (kosong = latest)
#   STATE_TOKEN  PAT buat restore world dari repo privat
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MC_DIR="${MC_DIR:-$HOME/mc}"
SRV="$MC_DIR/srv"
LOG="$MC_DIR/srv.log"

mkdir -p "$MC_DIR"
cd "$MC_DIR"

# ---- versi ----
if [ -n "${MCVER:-}" ]; then
  URL="https://www.minecraft.net/bedrockdedicatedserver/bin-linux/bedrock-server-${MCVER}.zip"
else
  URL=$(python3 "$REPO_DIR/scripts/mc_version.py")
fi
echo "URL=$URL"
curl -fL --http1.1 --retry 3 --retry-all-errors -A "Mozilla/5.0" -o bs.zip "$URL"
rm -rf "$SRV"
unzip -oq bs.zip -d "$SRV"

# ---- restore world ----
if [ -n "${STATE_TOKEN:-}" ]; then
  rm -rf world-repo
  if git clone -q --depth 1 \
      "https://x-access-token:${STATE_TOKEN}@github.com/Setyoanugrah/mcworld.git" world-repo 2>/dev/null; then
    if [ -d world-repo/worlds ]; then
      cp -a world-repo/worlds "$SRV/"
      echo ">> world restored dari repo"
    else
      echo ">> repo world kosong, bikin world baru"
    fi
  else
    echo ">> clone world gagal, bikin world baru"
  fi
fi

# ---- config ----
python3 "$REPO_DIR/scripts/mc_setup.py" "$SRV/server.properties"

# ---- start ----
cp "$REPO_DIR/scripts/mcctl.sh" "$MC_DIR/mcctl.sh"
chmod +x "$MC_DIR/mcctl.sh"
"$MC_DIR/mcctl.sh" start
