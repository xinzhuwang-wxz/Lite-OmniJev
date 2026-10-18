# Attribution and licenses

Lite-OmniJev adds decision LoRA weights, derived release formats, runtime code and documentation.
Project additions are distributed under CC-BY-NC-SA-4.0; see LICENSE.

## Base model

Gemma 4 E4B instruction model by Google DeepMind:
`google/gemma-4-E4B-it`, revision `ee0ef6023621cff504d758262d4e04895a5af4a2`.
The pinned upstream model card declares Apache-2.0. The license text is included at
[licenses/APACHE-2.0.txt](licenses/APACHE-2.0.txt).

Source: https://huggingface.co/google/gemma-4-E4B-it/tree/ee0ef6023621cff504d758262d4e04895a5af4a2
Official license: https://ai.google.dev/gemma/docs/gemma_4_license

Changes: decision LoRA training, FP32 merging, shared-key shard reconstruction, GGUF/MLX conversion,
input/readout gateway and product workbench. The source-bound receipts identify these derivatives.

## Demonstration media

- Screenshot: GUI-Odyssey dataset, `hflqf88888/GUIOdyssey`, revision
  `61632d0f3f4d51d7e9561ce4f84347dd06b2019d`, CC-BY-4.0. Derived RGB image, maximum side768.
  Source: https://huggingface.co/datasets/hflqf88888/GUIOdyssey
- Audio: `corneille_city01.wav`, 1demidegt2, CC0-1.0. Derived first6s mono16kHz clip.
  Source: https://freesound.org/people/1demidegt2/sounds/80685
- Video: CLEVRER train video00000, CC0. Derived first4s clip, eight model input frames.
  Source: https://clevrer.csail.mit.edu/
- Calendar question: project-authored example. Source clips, derivation hashes and inference recipes
  are recorded with the CPU demonstration traces.

Third-party runtime dependencies retain their upstream licenses. The service lock identifies their versions;
installation retrieves them from their package distributions. Data/training provenance is in DATA_CARD.md.
