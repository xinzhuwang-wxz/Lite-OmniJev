#!/usr/bin/env bash
set -euo pipefail

# Runs the official LiteRT-LM v0.9.0 linux_x86_64 host binary from the
# offline runtime cache without installing newer glibc/libstdc++ into the host.
# Defaults match the remote S6 preparation cache; override RUNTIME_BUILDS_ROOT
# when replaying in another checkout or artifact restore.

runtime_builds_root="${RUNTIME_BUILDS_ROOT:-/root/autodl-tmp/lite-omnijev/artifacts/runtime-builds}"
bin_dir="${LITERT_LM_BIN_DIR:-$runtime_builds_root/bin/litert-lm-v0.9.0-linux-x86_64}"
runtime_prefix="${LITERT_LM_RUNTIME_PREFIX:-$runtime_builds_root/toolchains/noble-runtime-prefix}"
loader="$runtime_prefix/usr/lib64/ld-linux-x86-64.so.2"
lib_path="$bin_dir:$runtime_prefix/usr/lib/x86_64-linux-gnu"
binary="$bin_dir/litert_lm_main"

if [[ ! -x "$binary" ]]; then
  printf 'missing LiteRT-LM binary: %s\n' "$binary" >&2
  exit 66
fi
if [[ ! -x "$loader" ]]; then
  printf 'missing private glibc loader: %s\n' "$loader" >&2
  exit 66
fi

exec env LD_LIBRARY_PATH="$lib_path" "$loader" --library-path "$lib_path" "$binary" "$@"
