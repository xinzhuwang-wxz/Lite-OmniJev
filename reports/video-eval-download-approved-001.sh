#!/usr/bin/env bash
set -u
cd /root/autodl-tmp/lite-omnijev
source infra/bootstrap/autodl-env.sh >/dev/null 2>&1
source /etc/network_turbo >/dev/null 2>&1
mkdir -p artifacts/data/research-eval/mvbench/230a2d4fac8900333c61754641c7a13e069ac9c6
rc=0
curl --fail --location --retry 3 --connect-timeout 30 --max-time 21600 --output artifacts/data/research-eval/mvbench/230a2d4fac8900333c61754641c7a13e069ac9c6/FunQA_test.zip https://huggingface.co/datasets/OpenGVLab/MVBench/resolve/230a2d4fac8900333c61754641c7a13e069ac9c6/video/FunQA_test.zip || rc=$?
printf "%s  %s\n" d6ef954652c2a1b48fb772d52c97da35dc7f8b54e5bdf4b43ba4f773ed726723 artifacts/data/research-eval/mvbench/230a2d4fac8900333c61754641c7a13e069ac9c6/FunQA_test.zip | sha256sum --check || rc=$?
curl --fail --location --retry 3 --connect-timeout 30 --max-time 21600 --output artifacts/data/research-eval/mvbench/230a2d4fac8900333c61754641c7a13e069ac9c6/perception.zip https://huggingface.co/datasets/OpenGVLab/MVBench/resolve/230a2d4fac8900333c61754641c7a13e069ac9c6/video/perception.zip || rc=$?
printf "%s  %s\n" 98fc197a90c875fb24f34fe89cb91e2788e1e7ac2459bf860ed62d9f07104b34 artifacts/data/research-eval/mvbench/230a2d4fac8900333c61754641c7a13e069ac9c6/perception.zip | sha256sum --check || rc=$?
printf "exit_code=%s\n" "$rc"
exit "$rc"
