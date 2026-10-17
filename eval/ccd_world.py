"""CCD world model: the symbolic scene state a CLEVRER-style program executes over.

A ``World`` is format-neutral: ground truth builds it via ``World.from_annotation``;
perception can build it from model atoms via ``World.from_tracks`` (in/out events are
derived from per-frame visibility) or by filling ``events`` directly.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Literal

from pydantic import Field, model_validator

from data.schema import Contract

Color = Literal["gray", "red", "blue", "green", "brown", "cyan", "purple", "yellow"]
Material = Literal["rubber", "metal"]
Shape = Literal["sphere", "cube", "cylinder"]
EventKind = Literal["in", "out", "collision"]

COLORS: tuple[str, ...] = ("gray", "red", "blue", "green", "brown", "cyan", "purple", "yellow")
MATERIALS: tuple[str, ...] = ("rubber", "metal")
SHAPES: tuple[str, ...] = ("sphere", "cube", "cylinder")
OBSERVED_FRAMES = 128


class SceneObject(Contract):
    object_id: int = Field(ge=0)
    color: Color
    material: Material
    shape: Shape


class TrackPoint(Contract):
    """Observed state of one object at one frame (floor-plane xy only)."""

    frame: int = Field(ge=0)
    in_view: bool
    x: float
    y: float
    vx: float
    vy: float


class Track(Contract):
    object_id: int = Field(ge=0)
    points: list[TrackPoint]

    @model_validator(mode="after")
    def _ordered(self) -> Track:
        frames = [p.frame for p in self.points]
        if frames != sorted(set(frames)):
            raise ValueError("track frames must be strictly increasing")
        return self


class Event(Contract):
    """An event. ``frame`` is None for a hypothetical event descriptor (``all_events``)."""

    kind: EventKind
    object_ids: tuple[int, ...]
    frame: int | None = None

    @model_validator(mode="after")
    def _arity(self) -> Event:
        want = 2 if self.kind == "collision" else 1
        if len(self.object_ids) != want or len(set(self.object_ids)) != want:
            raise ValueError(f"{self.kind} event needs {want} distinct object ids")
        if list(self.object_ids) != sorted(self.object_ids):
            raise ValueError("event object ids must be sorted")
        return self

    @property
    def key(self) -> tuple[str, tuple[int, ...]]:
        return (self.kind, self.object_ids)


def collision(a: int, b: int, frame: int | None = None) -> Event:
    return Event(kind="collision", object_ids=tuple(sorted((a, b))), frame=frame)


class Counterfactual(Contract):
    removed_object_id: int = Field(ge=0)
    events: list[Event]


class World(Contract):
    objects: list[SceneObject] = Field(min_length=1)
    tracks: list[Track] = Field(default_factory=list)
    events: list[Event] = Field(default_factory=list)
    unseen_events: list[Event] | None = None
    counterfactual: list[Counterfactual] | None = None
    horizon: int = Field(default=OBSERVED_FRAMES, ge=1)

    @model_validator(mode="after")
    def _ids(self) -> World:
        ids = [o.object_id for o in self.objects]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate object ids")
        known = set(ids)
        for t in self.tracks:
            if t.object_id not in known:
                raise ValueError(f"track for unknown object {t.object_id}")
        for e in [*self.events, *(self.unseen_events or [])]:
            if not set(e.object_ids) <= known:
                raise ValueError("event refers to unknown object")
        return self

    @property
    def object_ids(self) -> list[int]:
        return [o.object_id for o in self.objects]

    def obj(self, object_id: int) -> SceneObject:
        for o in self.objects:
            if o.object_id == object_id:
                return o
        raise KeyError(object_id)

    def track(self, object_id: int) -> Track | None:
        for t in self.tracks:
            if t.object_id == object_id:
                return t
        return None

    def counterfactual_events(self, object_id: int) -> list[Event]:
        for c in self.counterfactual or []:
            if c.removed_object_id == object_id:
                return c.events
        raise LookupError(f"no counterfactual for object {object_id}")

    @classmethod
    def from_tracks(
        cls,
        objects: Iterable[SceneObject],
        tracks: Iterable[Track],
        collisions: Iterable[Event],
        horizon: int = OBSERVED_FRAMES,
    ) -> World:
        """Derive ``in``/``out`` events from visibility and merge with collisions.

        ``in`` at frame f: the object is out of view at the previous sampled frame and in
        view at f (an object visible at its first sampled frame never "enters").
        ``out`` at frame f: in view at the previous sampled frame and out of view at f.
        """
        tracks = list(tracks)
        events: list[Event] = list(collisions)
        for t in tracks:
            for prev, cur in zip(t.points, t.points[1:], strict=False):
                if not prev.in_view and cur.in_view:
                    events.append(Event(kind="in", object_ids=(t.object_id,), frame=cur.frame))
                elif prev.in_view and not cur.in_view:
                    events.append(Event(kind="out", object_ids=(t.object_id,), frame=cur.frame))
        # Stable sort: same-frame collisions keep their given (causal) order.
        rank = {"in": 0, "collision": 1, "out": 2}
        events.sort(key=lambda e: (e.frame if e.frame is not None else -1, rank[e.kind]))
        return cls(objects=list(objects), tracks=tracks, events=events, horizon=horizon)

    @classmethod
    def from_annotation(
        cls, annotation: Mapping[str, Any], frames: Iterable[int] | None = None
    ) -> World:
        """Ground-truth world from a CLEVRER train/validation annotation json."""
        objects = [
            SceneObject(
                object_id=int(p["object_id"]),
                color=p["color"],
                material=p["material"],
                shape=p["shape"],
            )
            for p in annotation["object_property"]
        ]
        keep = None if frames is None else set(frames)
        per_obj: dict[int, list[TrackPoint]] = {o.object_id: [] for o in objects}
        trajectory = annotation["motion_trajectory"]
        for row in trajectory:
            f = int(row["frame_id"])
            if keep is not None and f not in keep:
                continue
            for o in row["objects"]:
                oid = int(o["object_id"])
                if oid not in per_obj:
                    continue
                per_obj[oid].append(
                    TrackPoint(
                        frame=f,
                        in_view=bool(o["inside_camera_view"]),
                        x=float(o["location"][0]),
                        y=float(o["location"][1]),
                        vx=float(o["velocity"][0]),
                        vy=float(o["velocity"][1]),
                    )
                )
        tracks = [
            Track(object_id=k, points=sorted(v, key=lambda p: p.frame)) for k, v in per_obj.items()
        ]
        collisions = [
            collision(int(c["object_ids"][0]), int(c["object_ids"][1]), int(c["frame_id"]))
            for c in annotation["collision"]
        ]
        return cls.from_tracks(objects, tracks, collisions, horizon=len(trajectory))
