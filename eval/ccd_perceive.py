"""Build a CCD ``World`` from sparse per-frame perception (presence + floor position).

Perception supplies, per sampled frame and object, ``(in_view, x, y)`` with ``x``/``y``
``None`` when no position was observed. This module turns that into what the executor and
physics need:

1. presence smoothing: optional majority filter over the sampled ``in_view`` flags;
2. side fits: for each gap between consecutive visible samples of an object, a line fitted
   on the samples before the gap and one on the samples after it;
3. collisions: for a pair visible across the same gap, the velocity changes ``dv_a`` and
   ``dv_b`` must look like an equal-mass impulse: ``J = (dv_b - dv_a) / 2`` large,
   ``dv_a + dv_b`` small, ``J`` pushing ``b`` away from ``a`` along the line of centres,
   and the centres about one radius sum apart. The contact time is where each object's
   before/after lines meet; per pair the strongest impulse within ``collision_nms`` wins;
4. dense tracks: a point for every frame ``0..horizon-1`` (physics anchors per frame);
   position and velocity from local line fits inside collision-free segments, rolled with
   the physics deceleration outside the visible span; ``in_view`` from the nearest sample,
   so entries/exits fall halfway between samples;
5. ``World.from_tracks`` derives in/out events and merges the collisions.

Velocities are in physics units: one frame moves ``FRAME_DT * velocity``. Every heuristic
is a field of ``PerceiveConfig``. Pure Python.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from eval.ccd_physics import FRAME_DT, PhysicsConfig
from eval.ccd_world import OBSERVED_FRAMES, Event, SceneObject, Track, TrackPoint, World, collision

Observation = tuple[bool, float | None, float | None]
Observations = Mapping[int, Mapping[int, Observation]]
_State = tuple[float, float, float, float]


@dataclass(frozen=True)
class PerceiveConfig:
    horizon: int = OBSERVED_FRAMES
    presence_window: int = 0  # half-width (samples) of the in_view majority filter; 0 = off
    fit_window: float = 1.0  # half-span (frames) of the local fit giving position/velocity
    fit_min_samples: int = 2  # widen the local fit to at least this many samples
    side_window: float = 8.0  # span (frames) of the before/after fits around a gap
    side_min_samples: int = 2  # widen each side fit to at least this many samples
    max_gap: float = 24.0  # no collision across a gap of unseen frames longer than this
    impulse_min: float = 0.08  # min |J| (velocity units) of a collision
    impulse_tol: float = 0.8  # |dv_a + dv_b| / 2 <= tol * |J|
    align_min: float = 0.5  # min cosine between J and the a->b centre line
    contact_margin: float = 0.25  # allowed centre distance beyond the radius sum
    collision_nms: float = 8.0  # one collision per pair within this many frames
    rearm_margin: float = 0.05  # a pair must be seen this far beyond reach to collide again
    frame_offset: float = 0.0  # added to the estimated contact time before rounding
    still_speed: float = 0.0  # speeds at or below this are reported as 0
    extrapolate_friction: bool = True  # roll unseen spans with the physics deceleration
    max_speed: float = 6.0  # backward rolls never speed up past this
    unseen_far: float = 100.0  # never-seen objects are parked this far off the floor
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)


@dataclass(frozen=True)
class _Sample:
    t: float
    x: float
    y: float


@dataclass(frozen=True)
class _Line:
    """Position ``p + s * (t - t0)`` with slope ``s`` in floor units per frame."""

    t0: float
    x: float
    y: float
    sx: float
    sy: float

    def at(self, t: float) -> tuple[float, float]:
        return (self.x + self.sx * (t - self.t0), self.y + self.sy * (t - self.t0))


@dataclass(frozen=True)
class _Kink:
    """Before/after lines of one object around one gap ``(t0, t1)``."""

    t0: float
    t1: float
    pre: _Line
    post: _Line

    @property
    def dv(self) -> tuple[float, float]:
        return (
            (self.post.sx - self.pre.sx) / FRAME_DT,
            (self.post.sy - self.pre.sy) / FRAME_DT,
        )

    def meet(self) -> tuple[float, float, float]:
        """Time in the gap where the lines are closest, and the position there."""
        p, q = self.pre, self.post
        dsx, dsy = p.sx - q.sx, p.sy - q.sy
        den = dsx * dsx + dsy * dsy
        if den == 0.0:
            t = (self.t0 + self.t1) / 2
        else:
            cx = p.x - p.sx * p.t0 - (q.x - q.sx * q.t0)
            cy = p.y - p.sy * p.t0 - (q.y - q.sy * q.t0)
            t = min(max(-(cx * dsx + cy * dsy) / den, self.t0), self.t1)
        (px, py), (qx, qy) = p.at(t), q.at(t)
        return t, (px + qx) / 2, (py + qy) / 2


def _fit(samples: Sequence[_Sample], t0: float) -> _Line:
    n = len(samples)
    mt = sum(s.t for s in samples) / n
    mx = sum(s.x for s in samples) / n
    my = sum(s.y for s in samples) / n
    var = sum((s.t - mt) ** 2 for s in samples)
    if var == 0.0:
        return _Line(t0, mx, my, 0.0, 0.0)
    sx = sum((s.t - mt) * (s.x - mx) for s in samples) / var
    sy = sum((s.t - mt) * (s.y - my) for s in samples) / var
    return _Line(t0, mx + sx * (t0 - mt), my + sy * (t0 - mt), sx, sy)


def _window(samples: Sequence[_Sample], t: float, half: float, least: int) -> list[_Sample]:
    near = sorted(samples, key=lambda s: (abs(s.t - t), s.t))
    chosen = [s for s in near if abs(s.t - t) <= half]
    if len(chosen) < least:
        chosen = near[:least]
    return sorted(chosen, key=lambda s: s.t)


def _side(
    samples: Sequence[_Sample], edge: int, step: int, span: float, least: int
) -> list[_Sample]:
    """Samples from index ``edge`` walking by ``step`` within ``span`` frames (>= ``least``)."""
    out: list[_Sample] = []
    i = edge
    while 0 <= i < len(samples):
        if len(out) >= least and abs(samples[i].t - samples[edge].t) > span:
            break
        out.append(samples[i])
        i += step
    return sorted(out, key=lambda s: s.t)


def _smooth_presence(flags: list[bool], half: int) -> list[bool]:
    if half <= 0:
        return flags
    out = []
    for i in range(len(flags)):
        win = flags[max(0, i - half) : i + half + 1]
        votes = 2 * sum(win)
        out.append(flags[i] if votes == len(win) else votes > len(win))
    return out


def _kinks(samples: list[_Sample], cfg: PerceiveConfig) -> dict[tuple[float, float], _Kink]:
    out: dict[tuple[float, float], _Kink] = {}
    for j in range(len(samples) - 1):
        a, b = samples[j], samples[j + 1]
        if b.t - a.t > cfg.max_gap:
            continue
        pre = _side(samples, j, -1, cfg.side_window, cfg.side_min_samples)
        post = _side(samples, j + 1, 1, cfg.side_window, cfg.side_min_samples)
        if len(pre) < 2 or len(post) < 2:
            continue
        out[(a.t, b.t)] = _Kink(a.t, b.t, _fit(pre, a.t), _fit(post, b.t))
    return out


def _separated(
    pa: list[_Sample], pb: list[_Sample], t0: float, t1: float, reach: float, cfg: PerceiveConfig
) -> bool:
    """Whether the pair is seen apart (beyond reach + ``rearm_margin``) strictly inside (t0, t1)."""
    where = {s.t: s for s in pb}
    for s in pa:
        o = where.get(s.t)
        if o is not None and t0 < s.t < t1:
            if math.hypot(o.x - s.x, o.y - s.y) > reach + cfg.rearm_margin:
                return True
    return False


def _collisions(
    kinks: dict[int, dict[tuple[float, float], _Kink]],
    paths: dict[int, list[_Sample]],
    shapes: dict[int, str],
    cfg: PerceiveConfig,
) -> list[Event]:
    radius = cfg.physics.radius
    found: list[tuple[float, float, int, int]] = []  # (time, |J|, a, b)
    ids = sorted(kinks)
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            reach = radius[shapes[a]] + radius[shapes[b]]
            hits: list[tuple[float, float]] = []
            for gap, ka in kinks[a].items():
                kb = kinks[b].get(gap)
                if kb is None:
                    continue
                (dax, day), (dbx, dby) = ka.dv, kb.dv
                jx, jy = (dbx - dax) / 2, (dby - day) / 2
                j = math.hypot(jx, jy)
                if j < cfg.impulse_min:
                    continue
                if math.hypot(dax + dbx, day + dby) / 2 > cfg.impulse_tol * j:
                    continue
                ta, ax, ay = ka.meet()
                tb, bx, by = kb.meet()
                t = (ta + tb) / 2
                ax, ay = ka.pre.at(t) if t <= ta else ka.post.at(t)
                bx, by = kb.pre.at(t) if t <= tb else kb.post.at(t)
                d = math.hypot(bx - ax, by - ay)
                if d - reach > cfg.contact_margin or d == 0.0:
                    continue
                if (jx * (bx - ax) + jy * (by - ay)) / (j * d) < cfg.align_min:
                    continue
                hits.append((t, j))
            kept: list[tuple[float, float]] = []
            for t, j in sorted(hits, key=lambda h: -h[1]):
                if all(abs(t - k[0]) >= cfg.collision_nms for k in kept):
                    kept.append((t, j))
            # A pair that never separated since its last collision is still in contact.
            prev: float | None = None
            for t, j in sorted(kept):
                if prev is None or _separated(paths[a], paths[b], prev, t, reach, cfg):
                    found.append((t, j, a, b))
                    prev = t
    events = []
    for t, _, a, b in sorted(found):
        frame = min(max(0, round(t + cfg.frame_offset)), cfg.horizon - 1)
        events.append(collision(a, b, frame))
    return events


def _roll(state: _State, shape: str, frames: int, cfg: PerceiveConfig) -> _State:
    """Move ``frames`` frames (negative = backwards) with the physics deceleration."""
    x, y, vx, vy = state
    p = cfg.physics
    step = 1 if frames > 0 else -1
    for _ in range(abs(frames)):
        v = math.hypot(vx, vy)
        if cfg.extrapolate_friction and v > 0.0:
            loss = p.mu[shape] + p.lin * v + p.quad * v * v
            if step > 0:
                scale = max(0.0, v - loss) / v
                vx, vy = vx * scale, vy * scale
                x, y = x + vx * FRAME_DT, y + vy * FRAME_DT
            else:
                x, y = x - vx * FRAME_DT, y - vy * FRAME_DT
                scale = min(v + loss, max(v, cfg.max_speed)) / v
                vx, vy = vx * scale, vy * scale
        else:
            x, y = x + step * vx * FRAME_DT, y + step * vy * FRAME_DT
    return (x, y, vx, vy)


def _state(segment: list[_Sample], t: float, cfg: PerceiveConfig) -> _State:
    line = _fit(_window(segment, t, cfg.fit_window, cfg.fit_min_samples), t)
    vx, vy = line.sx / FRAME_DT, line.sy / FRAME_DT
    if math.hypot(vx, vy) <= cfg.still_speed:
        vx = vy = 0.0
    return (line.x, line.y, vx, vy)


def _dense(
    oid: int,
    shape: str,
    flags: list[tuple[int, bool]],
    samples: list[_Sample],
    cuts: list[float],
    cfg: PerceiveConfig,
) -> Track:
    horizon = cfg.horizon
    states: list[_State]
    if not samples:
        far = cfg.unseen_far + 10.0 * oid
        states = [(far, far, 0.0, 0.0)] * horizon
    else:
        segments: list[list[_Sample]] = [[] for _ in range(len(cuts) + 1)]
        for s in samples:
            segments[sum(s.t > c for c in cuts)].append(s)
        filled = [s for s in segments if s]
        first, last = int(math.ceil(samples[0].t)), int(math.floor(samples[-1].t))
        states = []
        for f in range(first, last + 1):
            seg = segments[sum(f > c for c in cuts)]
            if not seg:  # no sample between two cuts: borrow the nearest segment
                seg = min(filled, key=lambda s: min(abs(p.t - f) for p in s))
            states.append(_state(seg, f, cfg))
        head = _state(filled[0], first, cfg)
        tail = _state(filled[-1], last, cfg)
        before = [_roll(head, shape, f - first, cfg) for f in range(first)]
        after = [_roll(tail, shape, f - last, cfg) for f in range(last + 1, horizon)]
        states = before + states + after
    points = []
    for f, (x, y, vx, vy) in enumerate(states):
        nearest = min(flags, key=lambda r: (abs(r[0] - f), r[0]))[1] if flags else False
        points.append(TrackPoint(frame=f, in_view=nearest, x=x, y=y, vx=vx, vy=vy))
    return Track(object_id=oid, points=points)


def world_from_observations(
    objects: Iterable[SceneObject],
    observations: Observations,
    frames: Iterable[int],
    cfg: PerceiveConfig | None = None,
) -> World:
    """World from sampled observations: ``observations[frame][object_id] = (in_view, x, y)``.

    A missing entry means "not in view". Positions with ``in_view`` False are ignored.
    """
    cfg = cfg or PerceiveConfig()
    objects = list(objects)
    frames = sorted(set(frames))
    shapes: dict[int, str] = {o.object_id: o.shape for o in objects}
    flags: dict[int, list[tuple[int, bool]]] = {}
    samples: dict[int, list[_Sample]] = {}
    for o in objects:
        oid = o.object_id
        seen = [observations.get(f, {}).get(oid, (False, None, None)) for f in frames]
        smooth = _smooth_presence([bool(s[0]) for s in seen], cfg.presence_window)
        flags[oid] = list(zip(frames, smooth, strict=True))
        samples[oid] = [
            _Sample(float(f), float(x), float(y))
            for f, vis, (_, x, y) in zip(frames, smooth, seen, strict=True)
            if vis and x is not None and y is not None
        ]
    kinks = {oid: _kinks(s, cfg) for oid, s in samples.items()}
    events = _collisions(kinks, samples, shapes, cfg)
    cuts: dict[int, list[float]] = {oid: [] for oid in shapes}
    for e in events:
        for oid in e.object_ids:
            cuts[oid].append(float(e.frame or 0) - 0.5)
    tracks = [
        _dense(oid, shapes[oid], flags[oid], samples[oid], sorted(cuts[oid]), cfg) for oid in shapes
    ]
    return World.from_tracks(objects, tracks, events, horizon=cfg.horizon)
