#!/bin/bash
# mc_backup.sh — backup folder worlds/ ke repo privat mcworld.
# Dijalankan sebagai runzzx1 (pemilik $HOME/mc), dipanggil berulang oleh workflow.
set -u

MC_DIR="${MC_DIR:-$HOME/mc}"
REPO="${WORLD_REPO:-Setyoanugrah/mcworld}"
[ -z "${STATE_TOKEN:-}" ] && { echo "no STATE_TOKEN, skip"; exit 0; }
[ -d "$MC_DIR/srv/worlds" ] || { echo "belum ada worlds/"; exit 0; }

cd "$MC_DIR" || exit 1
REMOTE="https://x-access-token:${STATE_TOKEN}@github.com/${REPO}.git"

rm -rf world-repo
if ! git clone -q "$REMOTE" world-repo 2>/dev/null; then
  mkdir -p world-repo
  cd world-repo || exit 1
  git init -q -b main
  git config user.email box@local
  git config user.name box
  git remote add origin "$REMOTE"
  cd ..
fi

rm -rf world-repo/worlds
cp -a srv/worlds world-repo/ || { echo "copy gagal"; exit 0; }

cd world-repo || exit 1
git add -A
if git commit -qm "world backup $(date -u +%FT%TZ)" 2>/dev/null; then
  git push -q origin HEAD:main && echo ">> world backed up"
else
  echo ">> no change"
fi
