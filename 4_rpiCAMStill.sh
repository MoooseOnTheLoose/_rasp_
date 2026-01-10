#!/usr/bin/env bash
set -euo pipefail
IMAGE_DIR="/media/user/disk/images"
FALLBACK_DIR="/home/user/images"
LOG_FILE="/var/log/rpicam/rpicam.log"
SEG_MS="2000"   # 2 seconds settle
MOUNTPOINT="/media/user/disk"
sudo mkdir -p "$(dirname "$LOG_FILE")" || true
mkdir -p "$IMAGE_DIR" "$FALLBACK_DIR"
session_ts="$(date +%Y%m%d_%H%M%S)"
if mountpoint -q "$MOUNTPOINT"; then
  out_dir="$IMAGE_DIR"
else
  out_dir="$FALLBACK_DIR"
fi
output_file="${out_dir}/image_${session_ts}.jpg"
{
  printf "\n=== Still Capture Started: %s ===\n" "$session_ts"
  rpicam-still --timeout "$SEG_MS" --nopreview -o "$output_file"
  end_ts="$(date +%Y%m%d_%H%M%S)"
  printf "=== Still Capture ended (session %s, ended %s) ===\n" "$session_ts" "$end_ts"
} >>"$LOG_FILE" 2>&1
