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
