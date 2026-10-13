from pathlib import Path

import pytest

from scripts.loop import claim, frontier, read_tickets, validate


def ticket(root: Path, number: str, depends: str = "None") -> None:
    folder = root / ".scratch/demo/issues"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{number}-task.md").write_text(
        f"ID: demo:{number}\nMilestone: M0\nCategory: enhancement\n"
        "Status: ready-for-agent\nExecution: open\nOwner: none\nWorktree: none\n"
        f"Blocked by: {depends}\n\n- [ ] Behavior verified\n"
    )


def test_claim_prevents_duplicate_and_blocked_work(tmp_path: Path) -> None:
    ticket(tmp_path, "01")
    ticket(tmp_path, "02", "01")
    (tmp_path / ".git").mkdir()
    assert frontier(read_tickets(tmp_path)) == ["demo:01"]
    claim(tmp_path, "demo:01", "alice", tmp_path)
    assert frontier(read_tickets(tmp_path)) == []
    with pytest.raises(ValueError, match="not unclaimed"):
        claim(tmp_path, "demo:01", "bob", tmp_path)
    with pytest.raises(ValueError, match="not unclaimed"):
        claim(tmp_path, "demo:02", "bob", tmp_path)


def test_invalid_dependency_and_cycles_fail_closed(tmp_path: Path) -> None:
    ticket(tmp_path, "01", "02")
    with pytest.raises(ValueError, match="missing dependency"):
        validate(read_tickets(tmp_path))
    ticket(tmp_path, "02", "01")
    with pytest.raises(ValueError, match="cycle"):
        validate(read_tickets(tmp_path))


def test_completion_requires_evidence_and_checks(tmp_path: Path) -> None:
    ticket(tmp_path, "01")
    path = tmp_path / ".scratch/demo/issues/01-task.md"
    path.write_text(path.read_text().replace("Execution: open", "Execution: done"))
    with pytest.raises(ValueError, match="without complete"):
        validate(read_tickets(tmp_path))
    path.write_text(path.read_text().replace("- [ ]", "- [x]") + "\nEvidence: reports/run.json\n")
    validate(read_tickets(tmp_path))
