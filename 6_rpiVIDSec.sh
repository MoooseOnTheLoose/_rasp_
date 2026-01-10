#!/usr/bin/env bash
set -euo pipefail
VIDEO_DIR="/media/user/disk/videos"
FALLBACK_DIR="/home/user/videos"
LOG_FILE="/var/log/rpicam/rpicam.log"
MOUNTPOINT="/media/user/disk"
SEG_MS="600000"      # 10 minutes per clip (milliseconds)
FPS="15"
BITRATE="2000000"    # 2 Mbps
MIN_FREE_GB="100"    # stop when free space drops below this
sudo mkdir -p "$(dirname "$LOG_FILE")" || true
mkdir -p "$VIDEO_DIR" "$FALLBACK_DIR"
session_ts="$(date +%Y%m%d_%H%M%S)"
if mountpoint -q "$MOUNTPOINT"; then
  out_dir="$VIDEO_DIR"
else
  out_dir="$FALLBACK_DIR"
fi
seq=0
cleanup_note() {
  {
    printf "=== Recording interrupted by user (Ctrl+C) ===\n"
  } >>"$LOG_FILE" 2>&1
  exit 130
}
trap cleanup_note INT
{
  printf "\n=== Vid-Sec Started: %s ===\n" "$session_ts"
  printf "DIR: %s\n" "$out_dir"
} >>"$LOG_FILE" 2>&1
while true; do
  # Free space in GB (integer)
  free_gb="$(df -BG --output=avail "$out_dir" | tail -n1 | tr -dc '0-9')"
  if [[ -z "${free_gb}" ]]; then free_gb=0; fi
  if (( free_gb < MIN_FREE_GB )); then
    {
      printf "\n=== STOP: low disk space (%s GB free < %s GB) ===\n" "$free_gb" "$MIN_FREE_GB"
    } >>"$LOG_FILE" 2>&1
    break
  fi
  clip_ts="$(date +%Y%m%d_%H%M%S)"
  mp4="${out_dir}/sec_${session_ts}_$(printf "%04d" "$seq")_${clip_ts}.mp4"
  cmd=( rpicam-vid
    --timeout "$SEG_MS"
    --nopreview
    --codec h264
    --framerate "$FPS"
    --bitrate "$BITRATE"
    --intra "$FPS"
    --inline
    -o "$mp4"
  )
  {
    printf "\nCMD: %s\n" "${cmd[*]}"
  } >>"$LOG_FILE" 2>&1

  "${cmd[@]}" >>"$LOG_FILE" 2>&1
  seq=$((seq + 1))
done
{
  end_ts="$(date +%Y%m%d_%H%M%S)"
  printf "=== Vid-Sec ended (session %s, ended %s) ===\n" "$session_ts" "$end_ts"
} >>"$LOG_FILE" 2>&1
