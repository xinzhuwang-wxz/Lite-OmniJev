#!/usr/bin/env bash
set -u
cd /root/autodl-tmp/lite-omnijev
source infra/bootstrap/autodl-env.sh >/dev/null 2>&1
source /etc/network_turbo >/dev/null 2>&1
rc=0
curl --fail --location --retry 3 --connect-timeout 30 --max-time 21600 --output artifacts/data/research-eval/mvbench/230a2d4fac8900333c61754641c7a13e069ac9c6/sta.zip https://huggingface.co/datasets/OpenGVLab/MVBench/resolve/230a2d4fac8900333c61754641c7a13e069ac9c6/video/sta.zip || rc=$?
printf "%s  %s\n" 4c27cbf62bbc1b0217b5183871f298e7a76b2a498dd4482365c617f25338de9a artifacts/data/research-eval/mvbench/230a2d4fac8900333c61754641c7a13e069ac9c6/sta.zip | sha256sum --check || rc=$?
printf "exit_code=%s\n" "$rc"
exit "$rc"
