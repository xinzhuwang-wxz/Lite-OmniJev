# Experimental motion coverage, CPU only

Status: mathematical helper and single-video CPU manifest; no loader, training,
service, inference-engine or GPU changes. No quality or novelty claim.

`motion_coverage_indices(scores, num_frames=16)` consumes one finite nonnegative
score per frame; frame i uses mean absolute difference from frame i-1 of decoded
64×64 grayscale thumbnails. Frame0 score is0. Keep frame0 and frameN-1. Partition
interior [1,N-1) into14 contiguous nonempty segments with floor boundaries,
and select the earliest maximum per segment. Output is16 strictly increasing,
unique indices. This retains temporal coverage rather than choosing global motion peaks.

`scripts/video_motion_manifest.py --video PATH --expected-sha256 SHA --output NEW.json`
accepts one explicit SHA-bound regular local file, preserving it unchanged. It
reuses the existing media decoder's file/pipe protocol and self-contained format
whitelist and its duration/FPS/nb_frames metadata convention. Additional fixed
bounds:256MiB source, positive FPS, ≤60seconds,16..7200 declared frames,
≤40million source pixels/frame. Unknown frame metadata is rejected.

FFmpeg decodes all frames into64×64gray with one decoder/filter/output thread.
Raw output goes to a temporary file, not an unbounded pipe capture. A7201-frame
output cap gives a29,495,296-byte bound and detects extra frames beyond permitted
metadata; decoded bytes must equal declared frame count×4096. Temporary pixels
are deleted. Only adjacent mean differences, source SHA, FPS/count/duration,
uniform and adaptive indices, nominal index/FPS times, code hashes and CPU timing
are recorded. Nominal timestamps do not claim exact VFR decoded PTS.

Uniform indices exactly retain current loader rule `round(i*(N-1)/15)`.
The manifest exclusively creates its output; failures leave a failed receipt.
CPU tests include hand-checked segments/ties/invalid scores, real synthetic40-frame
MP4 decoding, source preservation, input whitelist and metadata/decode bounds.

Any later paired public-dev50 temporal experiment must hold16frames and every
other input/model/processor setting fixed. Production integration waits for
benefit evidence. The previous CE+.1pair route was stopped after prediction ties
and increased memory; this is a separate sampling hypothesis, not another sweep.


## Frozen dev50 experiment (before GPU)

Use CE0 working adapter `4b8e422f3ca29e863999d2cd713bb9fd3f0a46e5360faa259a3bb8a4a9814bed`
from `artifacts/s4-claim-pair-ce0-train-001/adapter`. This supersedes the proposed
coverage parent before any GPU sampling evidence; both policies use CE0.
Select the first50 samples by SHA256(`42:s4-motion-coverage:`+id) from accepted
original video dev500 SHA79ed30be… whose whitespace-split instruction contains
before/after/when/while. Require all500source rows to be dev, unique and valid.

`.scratch/s4/evaluate-video-motion-coverage.py` defaults to CPU preparation,
≤120seconds, deduplicating per-video manifests. GPU requires both `--execute`
and one `--policy uniform|motion`; each fresh policy directory has a180second
alarm, progress at25 and50originals, and a failure receipt. Run a fresh uniform50
reference with the same CE0, not an older adapter's evidence. Both use16frames,
70video soft tokens/frame,4096max prompt tokens, BF16, one pass, the same official
processor/template/readout, and the same pinned input videos. Only indices vary.
The explicit process-local baseline loader hook calls real `load_media` with
`frame_indices`; production defaults remain uniform and unchanged.

CPU validation:91targeted local tests passed; one RGB-parity test skipped for
missing local Pillow. A real remote existing-runtime synthetic40frame CPU test confirmed
implicit versus explicit-uniform RGB bytes and metadata exactly equal, explicit
alternative indices retained FPS/count/duration, and source SHA unchanged.
No real dev50 preparation or GPU evaluation has run in this slice.
