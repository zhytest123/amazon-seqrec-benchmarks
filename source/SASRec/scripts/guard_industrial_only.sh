#!/usr/bin/env bash
set -euo pipefail

master_pid="${1:?master pid required}"

while ps -p "${master_pid}" >/dev/null 2>&1; do
  bad_jobs="$(pgrep -af -- '--category Musical_Instruments|--category CDs_and_Vinyl' || true)"
  if [ -n "${bad_jobs}" ]; then
    echo "$(date -Is) blocking non-Industrial job:"
    echo "${bad_jobs}"
    pkill -f -- '--category Musical_Instruments|--category CDs_and_Vinyl' || true
  fi
  sleep 10
done

