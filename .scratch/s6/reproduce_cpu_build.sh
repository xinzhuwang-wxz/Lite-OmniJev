#!/usr/bin/env bash
# Existing cache only; no model/network/dependency installation or GPU backend.
set -euo pipefail
archive=${1:?verified cached llama.cpp-v0.6.0.tar.gz}
workspace=${2:?new absolute workspace outside a Git checkout}
case "$workspace" in /*) ;; *) exit 2;; esac
[[ ! -e "$workspace" ]]
expected=09b36dba235fcac180efff18514da7038e65d8e5b9e4aefa59238468abfdec12
actual=$(shasum -a 256 "$archive" | cut -d ' ' -f1)
[[ "$actual" == "$expected" ]]
script_dir=$(cd -- "$(dirname -- "$0")" && pwd)
mkdir -p "$workspace/source" "$workspace/project"
# Hash-bound upstream archive; avoid inherited parent Git build provenance.
if git -C "$(dirname -- "$workspace")" rev-parse --show-toplevel >/dev/null 2>&1; then
  echo 'workspace must be outside a Git checkout' >&2
  exit 2
fi
tar -xzf "$archive" -C "$workspace/source"
cp "$script_dir/CMakeLists.txt" "$script_dir/raw_logits_probe.cpp" "$workspace/project/"
cmake -S "$workspace/project" -B "$workspace/build" \
  -DLLAMA_SOURCE="$workspace/source/llama.cpp-0.6.0" -DCMAKE_BUILD_TYPE=Release \
  >"$workspace/configure.log" 2>&1
cmake --build "$workspace/build" --target raw_logits_probe llama-mtmd-cli llama-quantize -j4 \
  >"$workspace/build.log" 2>&1
"$workspace/build/raw_logits_probe" --self-test >"$workspace/readout-self-test.json"
