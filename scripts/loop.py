"""Inspect and claim the local issue frontier; scheduling stays with the native goal agent."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

FIELDS = ("ID", "Milestone", "Category", "Status", "Execution", "Owner", "Worktree", "Blocked by")
ROLES = {"needs-triage", "needs-info", "ready-for-agent", "ready-for-human", "wontfix"}
STATES = {"open", "claimed", "blocked", "done"}


def read_tickets(root: Path) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for path in sorted((root / ".scratch").glob("*/issues/*.md")):
        body = path.read_text()
        item = {key: value.strip() for key, value in re.findall(r"^([\w ]+): (.*)$", body, re.M)}
        missing = set(FIELDS) - item.keys()
        if missing:
            raise ValueError(f"{path}: missing fields {sorted(missing)}")
        if item["ID"] in result:
            raise ValueError(f"Duplicate ID: {item['ID']}")
        item["path"] = str(path)
        item["body"] = body
        result[item["ID"]] = item
    if not result:
        raise ValueError("No local tickets found")
    return result


def blockers(item: dict[str, str]) -> list[str]:
    feature = item["ID"].split(":")[0]
    if item["Blocked by"] == "None":
        return []
    return [
        part if ":" in part else f"{feature}:{part}"
        for part in (x.strip() for x in item["Blocked by"].split(","))
    ]


def validate(items: dict[str, dict[str, str]]) -> None:
    for key, item in items.items():
        if item["Status"] not in ROLES or item["Execution"] not in STATES:
            raise ValueError(f"{key}: invalid state")
        if item["Category"] not in {"bug", "enhancement"}:
            raise ValueError(f"{key}: invalid category")
        if item["Execution"] == "claimed":
            if item["Owner"] == "none" or not Path(item["Worktree"]).is_absolute():
                raise ValueError(f"{key}: claim requires owner and absolute worktree")
        elif item["Owner"] != "none":
            raise ValueError(f"{key}: unclaimed ticket has owner")
        if "- [" not in item["body"]:
            raise ValueError(f"{key}: no acceptance checklist")
        for dep in blockers(item):
            if dep not in items:
                raise ValueError(f"{key}: missing dependency {dep}")
        if item["Execution"] == "done":
            if "- [ ]" in item["body"] or "Evidence:" not in item["body"]:
                raise ValueError(f"{key}: done without complete checklist/evidence")
            if any(items[dep]["Execution"] != "done" for dep in blockers(item)):
                raise ValueError(f"{key}: done before dependencies")
    visited: set[str] = set()
    active: set[str] = set()

    def visit(key: str) -> None:
        if key in active:
            raise ValueError(f"Dependency cycle: {key}")
        if key in visited:
            return
        active.add(key)
        for dep in blockers(items[key]):
            visit(dep)
        active.remove(key)
        visited.add(key)

    for key in items:
        visit(key)


def frontier(items: dict[str, dict[str, str]]) -> list[str]:
    return [
        key
        for key, item in items.items()
        if item["Status"] == "ready-for-agent"
        and item["Execution"] == "open"
        and item["Owner"] == "none"
        and all(items[dep]["Execution"] == "done" for dep in blockers(item))
    ]


@contextmanager
def queue_lock(root: Path) -> Iterator[None]:
    directory = root / ".cache"
    directory.mkdir(exist_ok=True)
    with (directory / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def atomic_write(path: Path, content: str) -> None:
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".ticket-")
    try:
        with os.fdopen(descriptor, "w") as output:
            output.write(content)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def claim(root: Path, ticket_id: str, owner: str, worktree: Path) -> None:
    if not owner or owner == "none" or "\n" in owner:
        raise ValueError("Owner must be a single nonempty identity")
    if not worktree.is_absolute() or not (worktree / ".git").exists():
        raise ValueError("Worktree must be an existing absolute Git checkout")
    with queue_lock(root):
        items = read_tickets(root)
        validate(items)
        if ticket_id not in frontier(items):
            raise ValueError(f"{ticket_id} is not unclaimed and ready")
        item = items[ticket_id]
        text = item["body"]
        for key, value in {
            "Execution": "claimed",
            "Owner": owner,
            "Worktree": str(worktree),
        }.items():
            text = "\n".join(
                f"{key}: {value}" if line.startswith(f"{key}: ") else line
                for line in text.split("\n")
            )
        atomic_write(Path(item["path"]), text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    sub.add_parser("validate")
    command = sub.add_parser("claim")
    command.add_argument("ticket")
    command.add_argument("--owner", required=True)
    command.add_argument("--worktree", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "claim":
            claim(args.root, args.ticket, args.owner, args.worktree)
        items = read_tickets(args.root)
        validate(items)
        print(
            json.dumps(
                {
                    "valid": True,
                    "tickets": len(items),
                    "frontier": frontier(items),
                    "claimed": [k for k, v in items.items() if v["Execution"] == "claimed"],
                    "blocked": [k for k, v in items.items() if v["Execution"] == "blocked"],
                }
            )
        )
        return 0
    except (ValueError, OSError) as error:
        print(json.dumps({"valid": False, "error": str(error)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
