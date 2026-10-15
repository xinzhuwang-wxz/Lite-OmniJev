"""Experimental CPU frame indices; no production loader or model integration."""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Real


def motion_coverage_indices(per_frame_motion: Sequence[float], num_frames: int = 16) -> list[int]:
    """Keep endpoints and the earliest highest-motion frame in each time segment.

    scores[i] describes the adjacent thumbnail difference of frame i from i-1;
    scores[0] is conventionally zero. Split interior indices [1, N-1) into
    num_frames-2 contiguous, nonempty floor-boundary segments. This preserves
    coverage, unlike global top-motion selection; it does not claim quality gains.
    """
    if type(num_frames) is not int or num_frames < 2:
        raise ValueError("num_frames must be an integer at least two")
    count = len(per_frame_motion)
    if count < num_frames:
        raise ValueError("video must contain at least num_frames frames")
    if any(
        isinstance(score, bool)
        or not isinstance(score, Real)
        or not math.isfinite(score)
        or score < 0
        for score in per_frame_motion
    ):
        raise ValueError("motion scores must be finite nonnegative real numbers")
    if num_frames == 2:
        return [0, count - 1]
    interior, segments = count - 2, num_frames - 2
    indices = [0]
    for segment in range(segments):
        start = 1 + segment * interior // segments
        stop = 1 + (segment + 1) * interior // segments
        indices.append(max(range(start, stop), key=lambda index: per_frame_motion[index]))
    return [*indices, count - 1]
