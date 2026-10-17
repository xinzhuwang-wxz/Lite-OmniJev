"""Stack interpreter for CLEVRER-style postfix programs over a ``World``.

Programs are postfix token lists: value tokens (``objects``, ``events``, attribute words,
``first``...) push; operators pop their arguments (in written order) and push a result.
Semantics were fitted against official CLEVRER train answers (see scripts/ccd_oracle_eval.py).

Operator semantics (stack arguments listed bottom→top):
- objects: all scene objects.  events: observed events.  all_events: every hypothetical
  event descriptor (each pair collision, each in/out, no frame).  unseen_events: events
  predicted after the observed horizon (needs ``World.unseen_events``).
- color/material/shape words, first/second/last, start/end, null: literal tokens.
- filter_color/material/shape(objs, word): objects with that attribute.
- unique(set): its single element; error if the set is not a singleton (events with the
  same kind+objects count as one; the earliest is kept).
- count(set) -> int; exist(set) -> yes/no; query_color/shape/material(obj) -> word.
- filter_in / filter_out(events, objs|obj): enter/exit events of those objects.
- filter_collision(events, objs|obj): collisions with at least one participant in objs.
- get_col_partner(collision, obj): the other participant.
- get_object(in/out event): its object.
- filter_order(events, first|second|last): k-th event by frame; same-frame ties keep the
  world's event order.
- filter_before / filter_after(events, event): events strictly before / after its frame.
- get_frame(event|start|end): the event's frame, or the first / last observed frame.
- filter_moving / filter_stationary(objs, frame|null): at a frame, in-view objects whose
  speed is above / at most ``moving_speed`` (0.02; out-of-view objects are neither); with
  null, moving at some in-view frame / stationary at every in-view frame.
- filter_ancestor(events, event): causal ancestors: every earlier event (frame, then event
  order) sharing a participant with the event, closed transitively; a participant's own
  ``in`` event always counts as earlier.
- belong_to(set): membership of the choice value (event by kind+objects, frame ignored;
  object by participation in any event of the set) in the set.
- negate(bool): logical not.
- get_counterfact(all_events, obj): events of the counterfactual world without obj.
"""

from __future__ import annotations

from dataclasses import dataclass

from eval.ccd_world import COLORS, MATERIALS, SHAPES, Event, World

ORDINALS = ("first", "second", "last")
TIME_TOKENS = ("start", "end", "null")


class ExecutionError(ValueError):
    """Program cannot be executed on this world (bad types, non-unique, missing data)."""


@dataclass(frozen=True)
class Objs:
    ids: tuple[int, ...]


@dataclass(frozen=True)
class Obj:
    id: int


@dataclass(frozen=True)
class Events:
    items: tuple[Event, ...]


@dataclass(frozen=True)
class OneEvent:
    event: Event


@dataclass(frozen=True)
class Frame:
    frame: int | None


@dataclass(frozen=True)
class Token:
    word: str


Value = Objs | Obj | Events | OneEvent | Frame | Token | bool | int | str


@dataclass(frozen=True)
class ExecConfig:
    moving_speed: float = 0.02


DEFAULT_CONFIG = ExecConfig()


def _ids(v: Value) -> tuple[int, ...]:
    if isinstance(v, Objs):
        return v.ids
    if isinstance(v, Obj):
        return (v.id,)
    raise ExecutionError(f"expected objects, got {type(v).__name__}")


def _events(v: Value) -> tuple[Event, ...]:
    if isinstance(v, Events):
        return v.items
    raise ExecutionError(f"expected events, got {type(v).__name__}")


def _event(v: Value) -> Event:
    if isinstance(v, OneEvent):
        return v.event
    raise ExecutionError(f"expected one event, got {type(v).__name__}")


def _token(v: Value, allowed: tuple[str, ...]) -> str:
    if isinstance(v, Token) and v.word in allowed:
        return v.word
    raise ExecutionError(f"expected one of {allowed}, got {v!r}")


def _frame_key(e: Event) -> int:
    return e.frame if e.frame is not None else 10**9


def _speeds(world: World, object_id: int) -> list[tuple[int, bool, float]]:
    track = world.track(object_id)
    if track is None:
        raise ExecutionError(f"no track for object {object_id}")
    return [(p.frame, p.in_view, (p.vx * p.vx + p.vy * p.vy) ** 0.5) for p in track.points]


def _moving(world: World, object_id: int, frame: int | None, cfg: ExecConfig) -> bool:
    speeds = _speeds(world, object_id)
    if not speeds:
        raise ExecutionError(f"empty track for object {object_id}")
    if frame is None:
        return any(s > cfg.moving_speed for _, vis, s in speeds if vis)
    nearest = min(speeds, key=lambda r: abs(r[0] - frame))
    return nearest[1] and nearest[2] > cfg.moving_speed


def _stationary(world: World, object_id: int, frame: int | None, cfg: ExecConfig) -> bool:
    if frame is None:
        speeds = _speeds(world, object_id)
        return all(s <= cfg.moving_speed for _, vis, s in speeds if vis)
    nearest = min(_speeds(world, object_id), key=lambda r: abs(r[0] - frame))
    return nearest[1] and nearest[2] <= cfg.moving_speed


def ancestors(events: tuple[Event, ...], target: Event) -> tuple[Event, ...]:
    """Earlier events sharing a participant with the target, closed transitively.

    "Earlier" is (frame, position in the event sequence), so same-frame collisions keep
    their causal order. A participant's own ``in`` event always precedes its later
    events (entry frames from visibility can lag the first contact by a frame or two).
    """
    if target.frame is None:
        raise ExecutionError("ancestor of a frameless event")

    def order(e: Event, pos: int) -> tuple[int, int]:
        return (-1, -1) if e.kind == "in" else (_frame_key(e), pos)

    indexed = [(order(e, i), e) for i, e in enumerate(events) if e.frame is not None]
    target_pos = next((i for i, e in enumerate(events) if e == target), len(events))
    chosen: list[tuple[tuple[int, int], Event]] = []
    frontier = [(order(target, target_pos), target)]
    taken: set[int] = set()
    while frontier:
        cur_order, cur = frontier.pop()
        if cur.kind == "in":
            continue
        for k, (o, e) in enumerate(indexed):
            if k in taken or o >= cur_order or not set(e.object_ids) & set(cur.object_ids):
                continue
            taken.add(k)
            chosen.append((o, e))
            frontier.append((o, e))
    return tuple(e for _, e in sorted(chosen, key=lambda r: (_frame_key(r[1]), r[0])))


def _all_events(world: World) -> Events:
    ids = world.object_ids
    items: list[Event] = []
    for i, a in enumerate(ids):
        items.append(Event(kind="in", object_ids=(a,)))
        items.append(Event(kind="out", object_ids=(a,)))
        for b in ids[i + 1 :]:
            items.append(Event(kind="collision", object_ids=tuple(sorted((a, b)))))
    return Events(tuple(items))


def _unique(v: Value) -> Value:
    if isinstance(v, Objs):
        if len(v.ids) != 1:
            raise ExecutionError(f"unique on {len(v.ids)} objects")
        return Obj(v.ids[0])
    if isinstance(v, Events):
        keys = {e.key for e in v.items}
        if len(keys) != 1:
            raise ExecutionError(f"unique on {len(v.items)} events")
        return OneEvent(min(v.items, key=_frame_key))
    raise ExecutionError(f"unique on {type(v).__name__}")


def _belongs(choice: Value, pool: tuple[Event, ...]) -> bool:
    if isinstance(choice, OneEvent):
        return any(e.key == choice.event.key for e in pool)
    if isinstance(choice, Obj):
        return any(choice.id in e.object_ids for e in pool)
    raise ExecutionError(f"belong_to on {type(choice).__name__}")


def execute(
    world: World,
    program: list[str],
    choice: Value | None = None,
    config: ExecConfig = DEFAULT_CONFIG,
) -> Value:
    """Run a postfix program; ``choice`` is the evaluated choice value for belong_to."""
    stack: list[Value] = []

    def pop() -> Value:
        if not stack:
            raise ExecutionError("stack underflow")
        return stack.pop()

    for tok in program:
        if tok == "objects":
            stack.append(Objs(tuple(world.object_ids)))
        elif tok == "events":
            stack.append(Events(tuple(world.events)))
        elif tok == "all_events":
            stack.append(_all_events(world))
        elif tok == "unseen_events":
            if world.unseen_events is None:
                raise ExecutionError("world has no unseen_events")
            stack.append(Events(tuple(world.unseen_events)))
        elif tok in COLORS or tok in MATERIALS or tok in SHAPES:
            stack.append(Token(tok))
        elif tok in ORDINALS or tok in TIME_TOKENS:
            stack.append(Frame(None) if tok == "null" else Token(tok))
        elif tok in ("filter_color", "filter_material", "filter_shape"):
            words = {"filter_color": COLORS, "filter_material": MATERIALS, "filter_shape": SHAPES}
            word = _token(pop(), words[tok])
            attr = tok.removeprefix("filter_")
            ids = _ids(pop())
            stack.append(Objs(tuple(i for i in ids if getattr(world.obj(i), attr) == word)))
        elif tok == "unique":
            stack.append(_unique(pop()))
        elif tok == "count":
            v = pop()
            stack.append(len(_events(v)) if isinstance(v, Events) else len(_ids(v)))
        elif tok == "exist":
            v = pop()
            stack.append(bool(_events(v)) if isinstance(v, Events) else bool(_ids(v)))
        elif tok in ("query_color", "query_material", "query_shape"):
            v = pop()
            if not isinstance(v, Obj):
                raise ExecutionError(f"{tok} needs one object")
            stack.append(str(getattr(world.obj(v.id), tok.removeprefix("query_"))))
        elif tok in ("filter_in", "filter_out"):
            members = set(_ids(pop()))
            kind = tok.removeprefix("filter_")
            evs = _events(pop())
            stack.append(
                Events(tuple(e for e in evs if e.kind == kind and e.object_ids[0] in members))
            )
        elif tok == "filter_collision":
            members = set(_ids(pop()))
            evs = _events(pop())
            stack.append(
                Events(
                    tuple(e for e in evs if e.kind == "collision" and set(e.object_ids) & members)
                )
            )
        elif tok == "get_col_partner":
            v = pop()
            if not isinstance(v, Obj):
                raise ExecutionError("get_col_partner needs one object")
            e = _event(pop())
            if e.kind != "collision" or v.id not in e.object_ids:
                raise ExecutionError("object is not in the collision")
            stack.append(Obj(next(i for i in e.object_ids if i != v.id)))
        elif tok == "get_object":
            e = _event(pop())
            if e.kind == "collision":
                raise ExecutionError("get_object on a collision")
            stack.append(Obj(e.object_ids[0]))
        elif tok == "filter_order":
            which = _token(pop(), ORDINALS)
            ordered = sorted(_events(pop()), key=_frame_key)
            index = {"first": 0, "second": 1, "last": -1}[which]
            if len(ordered) < (2 if which == "second" else 1):
                raise ExecutionError(f"filter_order {which} on {len(ordered)} events")
            stack.append(OneEvent(ordered[index]))
        elif tok in ("filter_before", "filter_after"):
            pivot = _event(pop()).frame
            if pivot is None:
                raise ExecutionError("temporal filter on frameless event")
            evs = _events(pop())
            if tok == "filter_before":
                keep = tuple(e for e in evs if e.frame is not None and e.frame < pivot)
            else:
                keep = tuple(e for e in evs if e.frame is not None and e.frame > pivot)
            stack.append(Events(keep))
        elif tok == "get_frame":
            v = pop()
            if isinstance(v, Token) and v.word == "start":
                stack.append(Frame(0))
            elif isinstance(v, Token) and v.word == "end":
                stack.append(Frame(world.horizon - 1))
            else:
                stack.append(Frame(_event(v).frame))
        elif tok in ("filter_moving", "filter_stationary"):
            f = pop()
            if not isinstance(f, Frame):
                raise ExecutionError(f"{tok} needs a frame")
            ids = _ids(pop())
            test = _moving if tok == "filter_moving" else _stationary
            stack.append(Objs(tuple(i for i in ids if test(world, i, f.frame, config))))
        elif tok == "filter_ancestor":
            target = _event(pop())
            stack.append(Events(ancestors(_events(pop()), target)))
        elif tok == "belong_to":
            if choice is None:
                raise ExecutionError("belong_to needs a choice value")
            stack.append(_belongs(choice, _events(pop())))
        elif tok == "negate":
            v = pop()
            if not isinstance(v, bool):
                raise ExecutionError("negate needs a boolean")
            stack.append(not v)
        elif tok == "get_counterfact":
            v = pop()
            if not isinstance(v, Obj):
                raise ExecutionError("get_counterfact needs one object")
            _events(pop())
            try:
                stack.append(Events(tuple(world.counterfactual_events(v.id))))
            except LookupError as exc:
                raise ExecutionError(str(exc)) from exc
        else:
            raise ExecutionError(f"unknown token {tok!r}")
    if len(stack) != 1:
        raise ExecutionError(f"program left {len(stack)} values")
    return stack[0]


def answer_open(world: World, program: list[str], config: ExecConfig = DEFAULT_CONFIG) -> str:
    """Answer an open (descriptive) question as the official answer string."""
    v = execute(world, program, config=config)
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, str):
        return v
    raise ExecutionError(f"program ended with {type(v).__name__}")


def answer_choice(
    world: World,
    question_program: list[str],
    choice_program: list[str],
    config: ExecConfig = DEFAULT_CONFIG,
) -> bool:
    """True when the choice is correct for the multiple-choice question."""
    value = execute(world, choice_program, config=config)
    if not isinstance(value, Obj | OneEvent):
        raise ExecutionError(f"choice program ended with {type(value).__name__}")
    result = execute(world, question_program, choice=value, config=config)
    if not isinstance(result, bool):
        raise ExecutionError("question program must end in a boolean")
    return result
