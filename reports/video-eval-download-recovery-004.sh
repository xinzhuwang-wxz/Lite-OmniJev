#!/usr/bin/env bash
set -euo pipefail
trap 'result=$?; printf "process_exit_code=%s\n" "$result"' EXIT
cd /root/autodl-tmp/lite-omnijev
source infra/bootstrap/autodl-env.sh >/dev/null 2>&1
source /etc/network_turbo >/dev/null 2>&1
exec 9>/tmp/omnijev-perception-download.lock
flock -n 9
final=artifacts/data/research-eval/mvbench/230a2d4fac8900333c61754641c7a13e069ac9c6/perception.zip
partial=artifacts/data/research-eval/mvbench/230a2d4fac8900333c61754641c7a13e069ac9c6/perception.zip.partial
expected_bytes=11199611859
expected_sha=98fc197a90c875fb24f34fe89cb91e2788e1e7ac2459bf860ed62d9f07104b34
if [[ -f "$final" && ! -f "$partial" ]]; then
  if [[ $(stat -c %s "$final") -eq "$expected_bytes" ]]; then
    printf '%s  %s
' "$expected_sha" "$final" | flock /tmp/omnijev-cpu-heavy.lock sha256sum --check
    exit 0
  fi
  mv "$final" "$partial"
fi
[[ ! -e "$final" ]]
printf 'resume_partial_bytes=%s
' "$(stat -c %s "$partial")"
last_exit=1
for attempt in 1 2 3 4; do
  before=$(stat -c %s "$partial")
  [[ "$before" -le "$expected_bytes" ]]
  if [[ "$before" -eq "$expected_bytes" ]]; then break; fi
  printf 'attempt=%s before_bytes=%s\n' "$attempt" "$before"
  if curl --fail --location --retry 0 --connect-timeout 30 --max-time 3600 --continue-at - --output "$partial" https://huggingface.co/datasets/OpenGVLab/MVBench/resolve/230a2d4fac8900333c61754641c7a13e069ac9c6/video/perception.zip; then
    last_exit=0
  else
    last_exit=$?
  fi
  after=$(stat -c %s "$partial")
  printf 'attempt=%s curl_exit=%s after_bytes=%s\n' "$attempt" "$last_exit" "$after"
  [[ "$after" -ge "$before" && "$after" -le "$expected_bytes" ]]
  if [[ "$after" -eq "$expected_bytes" ]]; then break; fi
  if [[ "$attempt" -lt 4 ]]; then sleep 5; fi
done
[[ $(stat -c %s "$partial") -eq "$expected_bytes" ]]
printf '%s  %s
' "$expected_sha" "$partial" | flock /tmp/omnijev-cpu-heavy.lock sha256sum --check
mv "$partial" "$final"
printf 'promoted_verified_file=%s
' "$final"
