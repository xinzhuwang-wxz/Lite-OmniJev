# Native CPU raw-logits client (preparation only)

Existing upstream llama.cpp v0.6.0 archive SHA256 `09b36dba235fcac180efff18514da7038e65d8e5b9e4aefa59238468abfdec12` is required; no download/install/model fetching in these scripts. Copying the already-cached archive from the project's remote was the only transfer in this run. No engine source is modified.

Actual M4/arm64 build outputs retained outside Git:

- `/tmp/lite-omnijev-s6-cpu-001/build/raw_logits_probe`
- `/tmp/lite-omnijev-s6-cpu-001/build/bin/llama-mtmd-cli`
- `/tmp/lite-omnijev-s6-cpu-001/build/bin/llama-quantize`

Build/config/source/binary hashes and exits: `reports/explore/edge-native-cpu-preparation-001.json`. Source archive is extracted outside Git so build metadata honestly says commit unknown; archive SHA is the provenance, not inherited parent Git. All GPU backends/Accelerate/BLAS are off; OpenSSL off avoids Homebrew dylibs. This CPU build is not a latency-optimized Metal release.

```sh
bash .scratch/s6/reproduce_cpu_build.sh /absolute/existing/llama.cpp-v0.6.0.tar.gz /tmp/new-s6-cpu-workspace
/tmp/lite-omnijev-s6-cpu-001/build/raw_logits_probe --self-test
```

The helper calls the **public** `llama_get_logits_ith(ctx,-1)` and uses `llama_vocab_n_tokens` for the complete unsampled row. No sampler or logit-bias/top-k path exists. It outputs selected raw logits, complete-vocabulary probabilities, legal+U joint probabilities, and answer probabilities conditional on excludingU. The U row deliberately has no conditional-answer field. No temperature is silently applied; uncalibrated/T=1 is explicit. Duplicate IDs, unknown labels, nonfinite full vocabulary and bad arguments reject. Fixture includes an unselected token holding half the full vocabulary mass, proving raw-vocab and legal denominators differ.

Future actual text inference requires a **locally present accepted E2B GGUF**, **official prepared token IDs**, and **generation-boundary verified complete legal+U IDs**:

```sh
/tmp/lite-omnijev-s6-cpu-001/build/raw_logits_probe --model "$E2B_GGUF" --tokens "$VERIFIED_PREFIX_TOKEN_IDS_CSV" --answers "$VERIFIED_LABEL_COLON_ID_CSV"
```

These environment variables are local file/token metadata placeholders, not credentials. There is no current E2B decision GGUF, so this model-forward command was not run. Caller must prove completeness/order/official mapping; this client does not render chat templates or certify arbitrary caller-provided answer sets. It only accepts≤4095 supplied prefix tokens and decodes once with4CPU threads. It is a minimal text C API client, **not an end-to-end six-modality gateway**.

For media, standard upstream `mtmd_tokenize[_from_parts]` + `mtmd_helper_eval_chunks` can populate the same llama context, then the same raw getter applies. Source/header and native mtmd CLI compile prove API availability, not actual E2B media behavior. Future integration is an application-level use of public APIs, not an engine patch or new classification head. Fixed16frames/order/timestamps, image1120 and audio normalization need sameinput verification; default CLI4FPS is not equivalent. Official template/thinking-disablement and calibration registry must remain bound to the final recipe. No fake multimedia rows or private datasets were used.

Native binary links only macOS system libc++/libSystem. No model load, model conversion, GPU, Metal, actual media inference, final calibration or product acceptance occurred. Initial build attempted unavailable llama-cli target under SERVER=OFF (exit2); standard llama-mtmd-cli target fixed the build without enabling a server. Final configure/build/help/version/selftest exit0. Full fresh script replay was not repeated; equivalent final manual commands were run, script syntax checked.
