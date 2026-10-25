#!/bin/bash
# mcctl.sh — start/stop/status server Bedrock (jalan sebagai user pemilik $HOME)
# Aman dari "pkill bunuh diri": pola pid-nya di-anchor ke ./bedrock_server
set -u

SRV="${MC_DIR:-$HOME/mc/srv}"
LOG="${MC_LOG:-$HOME/mc/srv.log}"
BIN="bedrock_server"

pid() { pgrep -f "^\./$BIN" 2>/dev/null || true; }

case "${1:-start}" in
  start)
    [ -x "$SRV/$BIN" ] || { echo "belum ada server di $SRV"; exit 1; }
    P=$(pid)
    if [ -n "$P" ]; then echo "sudah jalan pid=$P"; exit 0; fi
    cd "$SRV" || exit 1
    export LD_LIBRARY_PATH=.
    setsid ./"$BIN" >>"$LOG" 2>&1 < /dev/null &
    sleep 20
    echo "--- log ---"; tail -4 "$LOG"
    echo "--- udp ---"; ss -ulpn 2>/dev/null | grep -E "19132|19133" || echo "(belum terlihat)"
    ;;
  stop)
    P=$(pid)
    if [ -n "$P" ]; then
      kill "$P" && echo "stopped pid=$P"
      for i in $(seq 1 15); do pid >/dev/null || break; sleep 1; done
    else
      echo "tidak jalan"
    fi
    ;;
  status)
    P=$(pid); echo "pid: ${P:-tidak jalan}"
    ss -ulpn 2>/dev/null | grep -E "19132|19133" || true
    [ -f "$LOG" ] && tail -3 "$LOG"
    ;;
  *)
    echo "usage: mcctl.sh start|stop|status"; exit 2;;
esac
