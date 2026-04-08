"""Tests for the one-time data migration script."""

import os
from unittest.mock import patch

import pytest


@pytest.fixture
def fake_repo(tmp_path):
    """Create a fake repo layout with data/ and models/ dirs."""
    (tmp_path / "data").mkdir()
    (tmp_path / "models").mkdir()
    return tmp_path


def test_moves_old_file_to_new_location(fake_repo):
    """Basic case: old file exists, new does not."""
    from migrate_data import _pick_and_move

    old = fake_repo / "betting_history.csv"
    new = fake_repo / "data" / "betting_history.csv"
    old.write_text("date,platform,game\n2026-01-01,FanDuel,Duke vs UNC\n")

    moved, conflicts = [], []
    _pick_and_move(str(old), str(new), "betting_history.csv", "data/betting_history.csv", moved, conflicts)

    assert len(moved) == 1
    assert len(conflicts) == 0
    assert not old.exists()
    assert new.exists()
    assert "Duke vs UNC" in new.read_text()


def test_conflicts_when_both_exist(fake_repo):
    """Both files exist -- should flag conflict, touch neither."""
    from migrate_data import _pick_and_move

    old = fake_repo / "betting_history.csv"
    new = fake_repo / "data" / "betting_history.csv"

    old.write_text("date,platform,game\n2026-01-01,FanDuel,Duke vs UNC\n2026-01-02,Kalshi,MSU vs UM\n")
    new.write_text("date,platform,game\n")

    moved, conflicts = [], []
    _pick_and_move(str(old), str(new), "betting_history.csv", "data/betting_history.csv", moved, conflicts)

    assert len(moved) == 0
    assert len(conflicts) == 1
    # Both files should be untouched
    assert old.exists()
    assert new.exists()
    assert "Duke vs UNC" in old.read_text()


def test_skips_when_old_does_not_exist(fake_repo):
    """No old file -- nothing to do."""
    from migrate_data import _pick_and_move

    old = fake_repo / "betting_history.csv"
    new = fake_repo / "data" / "betting_history.csv"

    moved, conflicts = [], []
    _pick_and_move(str(old), str(new), "betting_history.csv", "data/betting_history.csv", moved, conflicts)

    assert len(moved) == 0
    assert len(conflicts) == 0


def test_full_migrate_moves_root_files(fake_repo):
    """End-to-end: migrate() moves files from root to data/."""
    from migrate_data import migrate

    (fake_repo / "betting_history.csv").write_text("real ledger\n")
    (fake_repo / "performance_log.csv").write_text("real perf\n")

    with patch("migrate_data.BASE_DIR", str(fake_repo)):
        result = migrate()

    assert result is True
    assert not (fake_repo / "betting_history.csv").exists()
    assert (fake_repo / "data" / "betting_history.csv").read_text() == "real ledger\n"
    assert not (fake_repo / "performance_log.csv").exists()
    assert (fake_repo / "data" / "performance_log.csv").read_text() == "real perf\n"


def test_full_migrate_returns_false_on_conflict(fake_repo):
    """migrate() returns False and leaves files alone when conflict exists."""
    from migrate_data import migrate

    (fake_repo / "betting_history.csv").write_text("old ledger\n")
    (fake_repo / "data" / "betting_history.csv").write_text("new ledger\n")

    with patch("migrate_data.BASE_DIR", str(fake_repo)):
        result = migrate()

    assert result is False
    # Both files untouched
    assert (fake_repo / "betting_history.csv").read_text() == "old ledger\n"
    assert (fake_repo / "data" / "betting_history.csv").read_text() == "new ledger\n"
