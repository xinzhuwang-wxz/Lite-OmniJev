# Rank expansion numerical-control preparation

Actual initial GPU parity failed exit1 in40.716seconds: twelve original input/mapping hashes matched, but maximum joint probability difference.03096713447, conditional difference.03096712751 and legal+Ulogit difference.125 exceed the unchanged1e-3probability bound. Source/derivative/transform hashes are bound in the actual receipt copied into the aggregate JSON. Do not train this initializer based on the algebraic equality.

The CPU transform preserves source A/B bits and scale, adding exactly zero B columns, so the real-arithmetic delta is unchanged. Finite precision can differ: rank changes the A output dimension16→64 and B reduction dimension16→64, potentially selecting different GEMM tiling/reduction even when adapter weights are FP32 and TF32disabled. Casting/intermediate bf16rounding can then propagate through decoder layers. BF16spacing is.125for values in[16,32); a single.125binary logit-difference shift near p=.5can change probability by about.03125. The observed numbers are compatible with such rounding, not proof that this is the cause. Historical run/load/backend drift must be separated first.

`probe-rank-runtime-control.py` loads one frozen base once and both standard PEFT adapters into that same process. It reuses the fixed12official prepared examples, verifies original IDs/gold/token mapping/full media-input hashes, model assets and exact before-compatible template/readout/preparation code, then performs source16→expanded64→source16 (36forwards total). It logs active adapter/dtypes/eval state and four aggregate comparisons: fresh16vs historical, fresh64vs fresh16, repeated16vs fresh16, repeated16vs historical. Each comparison includes raw legal+Ulogit maxima, joint and conditional probability maxima and changed prediction count. Repeat logit equality is explicit. This isolates a repeat-stable rank-dependent discrepancy from historical drift without treating the old historical source run as the only reference.

One process has a90second hard signal deadline and occupied-GPU refusal; root owns execution, no GPU was launched here. This uses only one of the maximum two allowed90second GPU arms. Diagnostic statuscompletedmeans the comparisons finished, while parity_validated independently requires fresh64vs fresh16joint AND conditional differences≤1e-3. training_authorizedremains false in every outcome. No tolerance was changed and no core module/trainer was edited.

CPU preflight completed exit0 in11.101seconds (`artifacts/s4-rank-runtime-control-cpu-001`); two CPU tests passed plus Ruff check/format. These validate difference aggregation and an optional alternate algebraic construction: zero added A rows with nonzero added B columns also preserves BA and permits gradients into new A. It still has rank64GEMM shapes, so it is not guaranteed to fix numerical parity; no alternate initializer or GPU arm was created. If fresh16is repeat-stable and rank64still genuinely differs, this alternative may justify the one remaining controlled arm only after root decision. A custom split-kernel path would change the standard PEFT execution contract and is not proposed here.

Root-only command:

```sh
cd /root/autodl-tmp/s4-claim-pair-code-001
PYTHONPATH=. /root/autodl-tmp/envs/before-gpu-teacher-job-final-001/bin/python /root/autodl-tmp/s4-native-event-cpu-001/probe-rank-runtime-control.py --initializer /root/autodl-tmp/s1-text-lora-001/artifacts/s4-rank64-initializer-001 --expansion-report-sha256 05e97013ac5d474866fdf208611bfef618bdfb65f5770c18f2924d21d37b2be7 --output /root/autodl-tmp/s1-text-lora-001/artifacts/s4-rank-runtime-control-gpu-001 --execute
```

Use the indicated source cwd; its four preparation/readout hashes match the actual before receipt. Raw input/question rows remain outside Git; aggregate comparisons only are exported. No compatibility, training, generalization or production claim follows from CPU preflight. Cost null. Fresh controlled GPU evidence is pending.
