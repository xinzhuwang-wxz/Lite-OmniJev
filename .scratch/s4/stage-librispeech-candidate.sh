#!/usr/bin/env bash
set -euo pipefail
source /etc/network_turbo >/dev/null 2>&1 || true
stage=/root/autodl-tmp/lite-omnijev/source-audit/librispeech-s4-candidate-001
mkdir -p "$stage"
curl -fsSL --max-time 60 https://www.openslr.org/12/ -o "$stage/source-page.html"
curl -fsSL --max-time 60 https://openslr.trmal.net/resources/12/md5sum.txt -o "$stage/md5sum.txt"
mkdir "$stage/mirror-002"
for split in dev-clean test-clean; do
  curl -fsSL --retry 1 --connect-timeout 20 --max-time 180 "https://openslr.magicdatatech.com/resources/12/$split.tar.gz" -o "$stage/mirror-002/$split.tar.gz" > "$stage/mirror-002/$split.log" 2>&1 &
done
wait
stage="$stage/mirror-002"
cd "$stage"
printf '%s\n' '42e2234ba48799c1f50f24a7926300a1  dev-clean.tar.gz' '32fa31d27d2e1cad72775fee3f4849a9  test-clean.tar.gz' | md5sum -c -
sha256sum dev-clean.tar.gz test-clean.tar.gz
