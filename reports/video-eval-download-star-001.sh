#!/usr/bin/env bash
set -u
cd /root/autodl-tmp/lite-omnijev
source infra/bootstrap/autodl-env.sh >/dev/null 2>&1
source /etc/network_turbo >/dev/null 2>&1
rc=0
curl --fail --location --retry 3 --connect-timeout 30 --max-time 21600 --output artifacts/data/research-eval/mvbench/230a2d4fac8900333c61754641c7a13e069ac9c6/star.zip https://huggingface.co/datasets/OpenGVLab/MVBench/resolve/230a2d4fac8900333c61754641c7a13e069ac9c6/video/star.zip || rc=$?
printf "%s  %s\n" bdb5f192a98f78e79e51da6ae801f8ba79d1d09c0d8c19b6e63105251cb17585 artifacts/data/research-eval/mvbench/230a2d4fac8900333c61754641c7a13e069ac9c6/star.zip | sha256sum --check || rc=$?
printf "exit_code=%s\n" "$rc"
exit "$rc"
