from __future__ import annotations

from eval.ccd_executor import answer_open
from eval.ccd_perceive import Observation, world_from_observations
from eval.ccd_physics import FRAME_DT
from eval.ccd_world import SceneObject

SPHERES = [
    SceneObject(object_id=0, color="red", material="rubber", shape="sphere"),
    SceneObject(object_id=1, color="blue", material="metal", shape="sphere"),
]
CONTACT = 1.8 / (2.0 * FRAME_DT)  # frames until two spheres 4.0 apart at speed 2 touch


def _head_on(t: float) -> tuple[float, float]:
    """Two spheres closing at speed 2 each; after contact they bounce back at 0.6."""
    if t <= CONTACT:
        d = 2.0 * FRAME_DT * t
        return -2.0 + d, 2.0 - d
    back = 0.6 * FRAME_DT * (t - CONTACT)
    return -0.2 - back, 0.2 + back


def _observe(frames: list[int]) -> dict[int, dict[int, Observation]]:
    out: dict[int, dict[int, Observation]] = {}
    for f in frames:
        a, b = _head_on(f)
        out[f] = {0: (True, a, 0.0), 1: (True, b, 0.0)}
    return out


def test_head_on_collision_recovered_at_the_right_frame() -> None:
    for frames in (list(range(128)), list(range(0, 128, 4))):
        world = world_from_observations(SPHERES, _observe(frames), frames)
        hits = [e for e in world.events if e.kind == "collision"]
        assert [e.object_ids for e in hits] == [(0, 1)]
        assert hits[0].frame is not None and abs(hits[0].frame - CONTACT) <= 2
        prog = ["events", "objects", "filter_collision", "count"]
        assert answer_open(world, prog) == "1"


def test_objects_passing_apart_do_not_collide() -> None:
    frames = list(range(0, 128, 4))
    obs = {f: {0: (True, -2.0 + 0.05 * f, 0.0), 1: (True, 2.0 - 0.05 * f, 1.0)} for f in frames}
    world = world_from_observations(SPHERES, obs, frames)
    assert not [e for e in world.events if e.kind == "collision"]


def test_entry_and_exit_derived_from_presence() -> None:
    frames = list(range(0, 128, 4))
    obs: dict[int, dict[int, Observation]] = {}
    for f in frames:
        enters = (True, -3.0 + 0.04 * f, -2.0) if f >= 40 else (False, None, None)
        leaves = (True, 1.0 + 0.04 * f, 2.0) if f <= 88 else (False, None, None)
        obs[f] = {0: enters, 1: leaves}
    world = world_from_observations(SPHERES, obs, frames)
    ins = [(e.object_ids, e.frame) for e in world.events if e.kind == "in"]
    outs = [(e.object_ids, e.frame) for e in world.events if e.kind == "out"]
    assert [i for i, _ in ins] == [(0,)] and 36 < (ins[0][1] or 0) <= 40
    assert [o for o, _ in outs] == [(1,)] and 88 < (outs[0][1] or 0) <= 92
    first_in = ["events", "objects", "filter_in", "unique", "get_object", "query_color"]
    assert answer_open(world, first_in) == "red"


def test_still_and_moving_objects_are_told_apart() -> None:
    frames = list(range(0, 128, 8))
    obs = {f: {0: (True, 0.5, 0.5), 1: (True, -2.0, -2.0 + 0.02 * f)} for f in frames}
    world = world_from_observations(SPHERES, obs, frames)
    still = ["objects", "null", "filter_stationary", "unique", "query_color"]
    moving = ["objects", "null", "filter_moving", "unique", "query_color"]
    assert (answer_open(world, still), answer_open(world, moving)) == ("red", "blue")


def test_never_seen_object_yields_no_events() -> None:
    frames = list(range(0, 128, 8))
    obs = {f: {0: (True, 0.5, 0.5)} for f in frames}  # object 1 is never reported
    world = world_from_observations(SPHERES, obs, frames)
    assert not world.events
