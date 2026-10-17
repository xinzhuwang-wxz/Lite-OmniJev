from __future__ import annotations

import pytest
from pydantic import ValidationError

from eval.ccd_executor import ExecutionError, answer_choice, answer_open
from eval.ccd_physics import simulate_world
from eval.ccd_world import Counterfactual, SceneObject, Track, TrackPoint, World, collision

RED_SPHERE = ["objects", "red", "filter_color", "sphere", "filter_shape", "unique"]
BLUE_CUBE = ["objects", "blue", "filter_color", "unique"]
GREEN = ["objects", "green", "filter_color", "unique"]
GRAY = ["objects", "gray", "filter_color", "unique"]


def _track(oid: int, frames: int, vis: str, speed: float = 0.0, x0: float = 0.0) -> Track:
    return Track(
        object_id=oid,
        points=[
            TrackPoint(
                frame=f, in_view=vis[f] == "1", x=x0 + 0.03 * speed * f, y=0.0, vx=speed, vy=0.0
            )
            for f in range(frames)
        ],
    )


def _world() -> World:
    # 0 red rubber sphere: present, moving; 1 blue metal cube: present, still;
    # 2 green rubber cylinder: enters at 2, exits at 7; 3 gray metal sphere: enters at 4.
    objects = [
        SceneObject(object_id=0, color="red", material="rubber", shape="sphere"),
        SceneObject(object_id=1, color="blue", material="metal", shape="cube"),
        SceneObject(object_id=2, color="green", material="rubber", shape="cylinder"),
        SceneObject(object_id=3, color="gray", material="metal", shape="sphere"),
    ]
    tracks = [
        _track(0, 10, "1111111111", speed=1.0),
        _track(1, 10, "1111111111"),
        _track(2, 10, "0011111000", speed=2.0, x0=-1),
        _track(3, 10, "0000111111", speed=1.0, x0=-3),
    ]
    collisions = [collision(0, 1, 3), collision(2, 1, 5), collision(1, 3, 5), collision(0, 3, 8)]
    return World.from_tracks(objects, tracks, collisions, horizon=10)


def test_attribute_filters_queries_count_and_exist() -> None:
    w = _world()
    assert answer_open(w, ["objects", "metal", "filter_material", "count"]) == "2"
    assert answer_open(w, [*RED_SPHERE, "query_material"]) == "rubber"
    assert (
        answer_open(
            w,
            [
                "objects",
                "metal",
                "filter_material",
                "cube",
                "filter_shape",
                "unique",
                "query_color",
            ],
        )
        == "blue"
    )
    assert answer_open(w, ["objects", "yellow", "filter_color", "exist"]) == "no"
    with pytest.raises(ExecutionError):
        answer_open(w, ["objects", "sphere", "filter_shape", "unique", "query_color"])


def test_entry_exit_events_order_and_temporal_filters() -> None:
    w = _world()
    assert answer_open(w, ["events", "objects", "filter_in", "count"]) == "2"
    assert (
        answer_open(
            w,
            [
                "events",
                "objects",
                "filter_in",
                "second",
                "filter_order",
                "get_object",
                "query_color",
            ],
        )
        == "gray"
    )
    assert (
        answer_open(w, ["events", "objects", "filter_out", "unique", "get_object", "query_shape"])
        == "cylinder"
    )
    after_gray_enters = ["events", "events", *GRAY, "filter_in", "unique", "filter_after"]
    assert answer_open(w, [*after_gray_enters, "objects", "filter_collision", "count"]) == "3"
    before = ["events", "events", *GREEN, "filter_out", "unique", "filter_before"]
    assert answer_open(w, [*before, "objects", "filter_collision", "count"]) == "3"


def test_collision_partner_and_ordinal() -> None:
    w = _world()
    first = ["events", *BLUE_CUBE, "filter_collision", "first", "filter_order"]
    assert answer_open(w, [*first, *BLUE_CUBE, "get_col_partner", "query_color"]) == "red"
    last = ["events", *BLUE_CUBE, "filter_collision", "last", "filter_order"]
    # Same-frame collisions keep their given order: (1,3) was listed after (2,1).
    assert answer_open(w, [*last, *BLUE_CUBE, "get_col_partner", "query_color"]) == "gray"


def test_moving_and_stationary_by_frame_and_whole_video() -> None:
    w = _world()
    assert answer_open(w, ["objects", "start", "get_frame", "filter_moving", "count"]) == "1"
    assert (
        answer_open(w, ["objects", "null", "filter_stationary", "unique", "query_shape"]) == "cube"
    )
    at_exit = [*GREEN, "filter_out", "unique", "get_frame"]
    prog = ["objects", "sphere", "filter_shape", "events", *at_exit, "filter_moving", "count"]
    assert answer_open(w, prog) == "2"
    # Out-of-view objects are neither moving nor stationary at a frame.
    assert answer_open(w, ["objects", "end", "get_frame", "filter_moving", "count"]) == "2"


def test_explanatory_ancestors_with_objects_events_and_negate() -> None:
    w = _world()
    target = [
        "events",
        "events",
        *RED_SPHERE,
        "filter_collision",
        "objects",
        "gray",
        "filter_color",
        "unique",
        "filter_collision",
        "unique",
        "filter_ancestor",
    ]
    responsible = [*target, "belong_to"]
    not_responsible = [*responsible, "negate"]
    green_entry = ["events", *GREEN, "filter_in", "unique"]
    assert answer_choice(w, responsible, green_entry)
    assert answer_choice(w, responsible, GREEN)
    cube_hits_red = [
        "events",
        *BLUE_CUBE,
        "filter_collision",
        *RED_SPHERE,
        "filter_collision",
        "unique",
    ]
    assert answer_choice(w, responsible, cube_hits_red)
    assert not answer_choice(w, not_responsible, cube_hits_red)


def test_counterfactual_and_predictive_membership() -> None:
    w = _world().model_copy(
        update={
            "unseen_events": [collision(2, 3, 12)],
            "counterfactual": [Counterfactual(removed_object_id=1, events=[collision(0, 3, 8)])],
        }
    )
    red_gray = ["all_events", *RED_SPHERE, "filter_collision", *GRAY, "filter_collision", "unique"]
    green_gray = ["all_events", *GREEN, "filter_collision", *GRAY, "filter_collision", "unique"]
    without_cube = ["all_events", *BLUE_CUBE, "get_counterfact", "belong_to"]
    assert answer_choice(w, without_cube, red_gray)
    assert answer_choice(w, [*without_cube, "negate"], green_gray)
    assert answer_choice(w, ["unseen_events", "belong_to"], green_gray)
    assert not answer_choice(w, ["unseen_events", "belong_to"], red_gray)
    with pytest.raises(ExecutionError):
        answer_choice(w, ["all_events", *RED_SPHERE, "get_counterfact", "belong_to"], green_gray)


def test_event_contract_rejects_bad_arity() -> None:
    with pytest.raises(ValidationError):
        World(
            objects=[SceneObject(object_id=0, color="red", material="metal", shape="cube")],
            events=[collision(0, 0, 1)],
        )


def _line_world(blocker: bool) -> World:
    """Disk 0 slides toward still disk 1; disk 2 (optional) sits in its path first."""
    objects = [
        SceneObject(object_id=0, color="red", material="rubber", shape="sphere"),
        SceneObject(object_id=1, color="blue", material="rubber", shape="sphere"),
        SceneObject(object_id=2, color="green", material="metal", shape="sphere"),
    ]

    def still(oid: int, x: float, y: float) -> Track:
        return Track(
            object_id=oid,
            points=[TrackPoint(frame=f, in_view=True, x=x, y=y, vx=0, vy=0) for f in range(4)],
        )

    mover = Track(
        object_id=0,
        points=[
            TrackPoint(frame=f, in_view=True, x=-2 + 0.07 * f, y=0, vx=2.5, vy=0) for f in range(4)
        ],
    )
    tracks = [mover, still(1, 1.5, 0.0), still(2, 0.0, 0.0 if blocker else 2.0)]
    return World.from_tracks(objects, tracks, [], horizon=4)


def test_physics_predicts_and_counterfactual_removes_collisions() -> None:
    w = simulate_world(_line_world(blocker=True))
    assert w.unseen_events is not None
    first = w.unseen_events[0]
    assert first.object_ids == (0, 2) and first.frame is not None and first.frame >= 4
    hit_blue = [
        "all_events",
        *RED_SPHERE,
        "filter_collision",
        *BLUE_CUBE,
        "filter_collision",
        "unique",
    ]
    without_green = ["all_events", *GREEN, "get_counterfact", "belong_to"]
    assert answer_choice(w, without_green, hit_blue)
    free = simulate_world(_line_world(blocker=False))
    assert free.unseen_events is not None
    assert [e.object_ids for e in free.unseen_events] == [(0, 1)]
