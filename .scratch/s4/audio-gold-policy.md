# S4 audio gold source policy — frozen before candidate evaluation

Decision 2026-10-15, within ADR0012 source-selection authority and the user's allowance for real-data-driven preparation. No change to the baseline, quality thresholds, original Clotho dev500, or original pool membership. No candidate scores were used.

## Fixed composition

- Calibration: 1400 distinct original Clotho-AQA questions +1600 distinct original LibriSpeech utterance questions =3000.
- Holdout: 500 Clotho +500 LibriSpeech =1000. Report the two strata separately as well as their fixed 50/50 aggregate.
- Existing Clotho dev500 is unchanged. Additional speech dev500 is a separate diagnostic/paired-baseline stratum, not a replacement for those questions or a way to inflate the six-cohort original dev quotas.
- Original environmental-audio gates remain mandatory. Speech must also receive same-input teacher/base measurements and separate quality/calibration/coverage reporting. The S4 holdout/dev gap is checked per stratum; aggregate speech performance cannot offset a failed environmental stratum. No whole-audio pass from speech-only results.
- Any future adjustment requires a written decision before evaluation; no score-based mixture selection.

## Source and identity rules

Clotho: original three-unanimous yes/no answers, approved CC0/BY media and MIT questions; preserve all current train/dev/calibration/holdout membership and crossmodal aliases. Extend existing calibration/holdout families with distinct original questions. Among unallocated families, fixed-seed hash order plus subset-sum selects exactly the missing holdout question capacity, with whole families assigned together. Remaining untouched families are calibration candidates. Keep every previously reserved item; trim surplus new calibration questions by fixed hash. Variants and changed options do not add items.

LibriSpeech: official dev-clean/test-clean archive SHA pins in s4-librispeech-source-audit-001.json; CC BY4.0 as declared by https://www.openslr.org/12/. Original transcript is gold. One utterance gives one four-option transcription-match Choice question, with deterministic answer ordering and distinct distractors from its assigned pool only. These are read-English speech tasks, not environmental-sound questions. Entire speaker/book/transcript/media connected components stay within one pool, including unused members. Original archive split alone is insufficient isolation. Preserve original and derived media hashes and source attribution. Fully decode selected audio; duration <=30 seconds, 16kHz mono.

The capacity reports establish feasibility only: Clotho 1916 original available questions, including existing pools; 790 available media fully decoded. Speech 5283 deduplicated eligible utterances in70 components (headers only so far). Neither is final admission, permission evidence, or proof against base-model pretraining exposure.
