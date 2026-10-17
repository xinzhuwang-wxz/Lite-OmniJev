"""Deterministic 2-D floor physics for CLEVRER-style predictive and counterfactual events.

Objects are disks on the floor plane (radius by shape) that slide with a speed-dependent
deceleration ``mu[shape] + lin*v + quad*v^2`` per frame and collide with equal-mass
impulses (restitution ``restitution``). Constants were fitted to CLEVRER train
trajectories and tuned on a DEV subset (see scripts/ccd_oracle_eval.py).

- Predictive: start from the last observed state and roll forward; collisions that
  start after the observed horizon become ``World.unseen_events``.
- Counterfactual: remove one object and re-simulate from frame 0. Objects stay anchored
  to their observed tracks until the change can reach them (an observed collision with
  the removed or an already-diverged object, or a simulated contact with a diverged
  object); from then on they are simulated. This keeps unaffected events exact.
Pure Python (6-ish objects x ~200 frames); no numpy dependency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from eval.ccd_world import Counterfactual, Event, TrackPoint, World, collision

FRAME_DT = 0.02799  # seconds of motion per annotated frame (fitted |dx| / |v|)


def _default_radius() -> dict[str, float]:
    return {"sphere": 0.2, "cylinder": 0.2, "cube": 0.235}


def _default_mu() -> dict[str, float]:
    return {"sphere": 0.0002, "cylinder": 0.0083, "cube": 0.0083}


@dataclass(frozen=True)
class PhysicsConfig:
    radius: dict[str, float] = field(default_factory=_default_radius)
    mu: dict[str, float] = field(default_factory=_default_mu)  # speed loss per frame at v~0
    lin: float = 0.0015  # extra speed loss per frame per unit speed
    quad: float = 0.0011  # extra speed loss per frame per unit speed^2
    restitution: float = 0.3
    substeps: int = 4
    predict_frames: int = 80  # frames simulated after the observed horizon
    counterfactual_frames: int = 200  # total frames of a counterfactual world (fitted)
    anchored: bool = True  # counterfactual keeps unaffected objects on observed tracks
    rewind: int = 2  # frames before an observed collision used to free an object


@dataclass
class _Body:
    oid: int
    shape: str
    x: float
    y: float
    vx: float
    vy: float


def _state(p: TrackPoint) -> tuple[float, float, float, float]:
    return (p.x, p.y, p.vx, p.vy)


class _Sim:
    def __init__(self, cfg: PhysicsConfig, bodies: list[_Body]):
        self.cfg = cfg
        self.bodies = {b.oid: b for b in bodies}
        self.contacts: set[tuple[int, int]] = set()

    def _decel(self, b: _Body, dt_frames: float) -> None:
        v = math.hypot(b.vx, b.vy)
        if v == 0.0:
            return
        loss = (self.cfg.mu[b.shape] + self.cfg.lin * v + self.cfg.quad * v * v) * dt_frames
        scale = max(0.0, v - loss) / v
        b.vx *= scale
        b.vy *= scale

    def prime_contacts(self) -> None:
        """Treat already-touching pairs as ongoing contacts (no new event)."""
        ids = sorted(self.bodies)
        for i, a in enumerate(ids):
            for c in ids[i + 1 :]:
                if self._dist(a, c) < self._reach(a, c):
                    self.contacts.add((a, c))

    def _reach(self, a: int, c: int) -> float:
        r = self.cfg.radius
        return r[self.bodies[a].shape] + r[self.bodies[c].shape]

    def _dist(self, a: int, c: int) -> float:
        ba, bc = self.bodies[a], self.bodies[c]
        return math.hypot(bc.x - ba.x, bc.y - ba.y)

    def step(self, free: set[int]) -> list[tuple[int, int]]:
        """Advance one frame; only ``free`` bodies move/react. Returns new contacts."""
        new: list[tuple[int, int]] = []
        h = 1.0 / self.cfg.substeps
        ids = sorted(self.bodies)
        for _ in range(self.cfg.substeps):
            for oid in free:
                b = self.bodies[oid]
                b.x += b.vx * FRAME_DT * h
                b.y += b.vy * FRAME_DT * h
                self._decel(b, h)
            for i, a in enumerate(ids):
                for c in ids[i + 1 :]:
                    if a not in free and c not in free:
                        continue
                    pair = (a, c)
                    d = self._dist(a, c)
                    reach = self._reach(a, c)
                    if d >= reach:
                        if d > reach + 0.02:
                            self.contacts.discard(pair)
                        continue
                    if self._impulse(a, c, d, free) and pair not in self.contacts:
                        new.append(pair)
                    self.contacts.add(pair)
        return new

    def _impulse(self, a: int, c: int, d: float, free: set[int]) -> bool:
        ba, bc = self.bodies[a], self.bodies[c]
        if d == 0.0:
            return False
        nx, ny = (bc.x - ba.x) / d, (bc.y - ba.y) / d
        un = (bc.vx - ba.vx) * nx + (bc.vy - ba.vy) * ny
        if un >= 0:
            return False
        ma = 1.0 if a in free else math.inf
        mc = 1.0 if c in free else math.inf
        inv = (1 / ma) + (1 / mc)
        j = -(1 + self.cfg.restitution) * un / inv
        if a in free:
            ba.vx -= j * nx / ma
            ba.vy -= j * ny / ma
        if c in free:
            bc.vx += j * nx / mc
            bc.vy += j * ny / mc
        return True


def _points(world: World) -> dict[int, dict[int, TrackPoint]]:
    return {t.object_id: {p.frame: p for p in t.points} for t in world.tracks}


def _bodies(world: World, states: dict[int, tuple[float, float, float, float]]) -> list[_Body]:
    return [_Body(oid, world.obj(oid).shape, *states[oid]) for oid in sorted(states)]


def predict(world: World, cfg: PhysicsConfig) -> list[Event]:
    """Collisions that start after the last observed frame."""
    pts = _points(world)
    last = min(max(frames) for frames in pts.values())
    start = {oid: _state(frames[last]) for oid, frames in pts.items()}
    sim = _Sim(cfg, _bodies(world, start))
    sim.prime_contacts()
    free = set(start)
    events: list[Event] = []
    for f in range(last + 1, last + 1 + cfg.predict_frames):
        for a, c in sim.step(free):
            if f >= world.horizon:
                events.append(collision(a, c, f))
    return events


def _observed_at(
    pts: dict[int, TrackPoint], frame: int
) -> tuple[float, float, float, float] | None:
    p = pts.get(frame)
    return None if p is None else _state(p)


def counterfactual(world: World, removed: int, cfg: PhysicsConfig) -> list[Event]:
    """Collisions of the world re-simulated from frame 0 without ``removed``."""
    pts = _points(world)
    first = max(min(frames) for frames in pts.values())
    ids = [oid for oid in sorted(pts) if oid != removed]
    sim = _Sim(cfg, _bodies(world, {oid: _state(pts[oid][first]) for oid in ids}))
    sim.prime_contacts()
    free: set[int] = set() if cfg.anchored else set(ids)
    observed: dict[int, list[tuple[int, int]]] = {}
    for e in world.events:
        if e.kind == "collision" and e.frame is not None:
            pair = (e.object_ids[0], e.object_ids[1])
            observed.setdefault(e.frame, []).append(pair)
    events: list[Event] = []
    for f in range(first + 1, cfg.counterfactual_frames):
        anchored = [oid for oid in ids if oid not in free]
        for a, c in observed.get(f, []):
            if a in free or c in free or removed in (a, c):
                for oid in (a, c):
                    if oid in anchored and oid not in free:
                        _release(sim, pts[oid], oid, f, cfg)
                        free.add(oid)
        anchored = [oid for oid in ids if oid not in free]
        for oid in anchored:
            s = _observed_at(pts[oid], f)
            if s is None:
                _release(sim, pts[oid], oid, f, cfg)
                free.add(oid)
                continue
            b = sim.bodies[oid]
            b.x, b.y, b.vx, b.vy = s
        for a, c in observed.get(f, []):
            if a not in free and c not in free and removed not in (a, c):
                events.append(collision(a, c, f))
                sim.contacts.add((a, c))
        for a, c in sim.step(free) if free else []:
            events.append(collision(a, c, f))
            for oid in (a, c):
                free.add(oid)
    return events


def _release(
    sim: _Sim, pts: dict[int, TrackPoint], oid: int, frame: int, cfg: PhysicsConfig
) -> None:
    """Free an anchored body: restart from a pre-collision observed state, rolled forward."""
    base = frame - cfg.rewind
    while base > 0 and base not in pts:
        base -= 1
    if base not in pts:
        return
    b = sim.bodies[oid]
    b.x, b.y, b.vx, b.vy = _state(pts[base])
    for _ in range(frame - 1 - base):
        sim._decel(b, 1.0)
        b.x += b.vx * FRAME_DT
        b.y += b.vy * FRAME_DT


def simulate_world(world: World, cfg: PhysicsConfig | None = None) -> World:
    """Return a copy of ``world`` with ``unseen_events`` and all counterfactuals filled."""
    cfg = cfg or PhysicsConfig()
    if not world.tracks:
        raise ValueError("physics needs object tracks")
    return world.model_copy(
        update={
            "unseen_events": predict(world, cfg),
            "counterfactual": [
                Counterfactual(removed_object_id=oid, events=counterfactual(world, oid, cfg))
                for oid in world.object_ids
            ],
        }
    )
