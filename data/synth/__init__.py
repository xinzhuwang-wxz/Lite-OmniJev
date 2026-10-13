"""Small deterministic program-truth fixtures, never model-quality evidence."""

import hashlib
import random

from data.schema import Pool, Sample


def generate_samples(count: int = 12, seed: int = 0, pool: Pool = "train") -> list[Sample]:
    if type(count) is not int or count < 1:
        raise ValueError("count must be a positive integer")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    rng = random.Random(seed)
    samples = []
    for i in range(count):
        left, right = rng.randrange(100), rng.randrange(100)
        state = f"There are {left} red balls and {right} blue balls."
        # Identity tracks the underlying facts, independent of seed or destination pool.
        family = hashlib.sha256(state.encode()).hexdigest()
        samples.append(
            Sample.model_validate(
                {
                    "id": f"program-{seed}-{i}",
                    "prefix": {"state": state},
                    "questions": [
                        {
                            "qid": "compare",
                            "type": "choice",
                            "instruction": "Which color has more balls?",
                            "options": ["red", "blue", "equal"],
                            "target": {
                                "kind": "hard",
                                "value": "A" if left > right else "B" if right > left else "C",
                                "source": "program",
                            },
                        },
                        {
                            "qid": "positive",
                            "type": "noul",
                            "instruction": "Are there more red than blue balls?",
                            "target": {
                                "kind": "hard",
                                "value": "yes" if left > right else "no",
                                "source": "program",
                            },
                        },
                        {
                            "qid": "score",
                            "type": "score",
                            "instruction": "How many colors have at least one ball?",
                            "score_levels": 3,
                            "target": {
                                "kind": "hard",
                                "value": int(left > 0) + int(right > 0),
                                "source": "program",
                            },
                        },
                        {
                            "qid": "unknown",
                            "type": "noul",
                            "instruction": "Are the balls made of wood?",
                            "target": {"kind": "insufficient"},
                        },
                    ],
                    "meta": {
                        "modality": ["text"],
                        "source_dataset": "program-balls",
                        "source_revision": "v1",
                        "source_sha256": family,
                        "license": "internal",
                        "usage": "commercial",
                        "pool": pool,
                        "family_id": family,
                        "data_version": "program-v1",
                    },
                }
            )
        )
    return samples
