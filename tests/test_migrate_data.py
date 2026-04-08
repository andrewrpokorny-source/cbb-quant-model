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

    moved, skipped = [], []
    _pick_and_move(str(old), str(new), "betting_history.csv", "data/betting_history.csv", moved, skipped)

    assert len(moved) == 1
    assert not old.exists()
    assert new.exists()
    assert "Duke vs UNC" in new.read_text()


def test_replaces_smaller_destination_with_real_ledger(fake_repo):
    """Reviewer scenario: bot created empty new file, real ledger is at old path."""
    from migrate_data import _pick_and_move

    old = fake_repo / "betting_history.csv"
    new = fake_repo / "data" / "betting_history.csv"

    # Real ledger at old location
    old.write_text("date,platform,game\n2026-01-01,FanDuel,Duke vs UNC\n2026-01-02,Kalshi,MSU vs UM\n")
    # Empty file created by bot at new location
    new.write_text("date,platform,game\n")

    moved, skipped = [], []
    _pick_and_move(str(old), str(new), "betting_history.csv", "data/betting_history.csv", moved, skipped)

    assert len(moved) == 1
    assert "replaced smaller destination" in moved[0][2]
    assert not old.exists()
    assert "Duke vs UNC" in new.read_text()
    assert "MSU vs UM" in new.read_text()


def test_skips_when_old_does_not_exist(fake_repo):
    """No old file -- nothing to do."""
    from migrate_data import _pick_and_move

    old = fake_repo / "betting_history.csv"
    new = fake_repo / "data" / "betting_history.csv"

    moved, skipped = [], []
    _pick_and_move(str(old), str(new), "betting_history.csv", "data/betting_history.csv", moved, skipped)

    assert len(moved) == 0
    assert len(skipped) == 0


def test_removes_old_when_destination_is_larger(fake_repo):
    """New file is the real one, old is stale -- remove old."""
    from migrate_data import _pick_and_move

    old = fake_repo / "betting_history.csv"
    new = fake_repo / "data" / "betting_history.csv"

    old.write_text("date\n")
    new.write_text("date,platform,game\n2026-01-01,FanDuel,Duke vs UNC\n")

    moved, skipped = [], []
    _pick_and_move(str(old), str(new), "betting_history.csv", "data/betting_history.csv", moved, skipped)

    assert len(skipped) == 1
    assert "destination is larger" in skipped[0][1]
    assert not old.exists()
    assert new.exists()


def test_full_migrate_moves_root_files(fake_repo):
    """End-to-end: migrate() moves files from root to data/."""
    from migrate_data import migrate

    (fake_repo / "betting_history.csv").write_text("real ledger\n")
    (fake_repo / "performance_log.csv").write_text("real perf\n")

    with patch("migrate_data.BASE_DIR", str(fake_repo)):
        migrate()

    assert not (fake_repo / "betting_history.csv").exists()
    assert (fake_repo / "data" / "betting_history.csv").read_text() == "real ledger\n"
    assert not (fake_repo / "performance_log.csv").exists()
    assert (fake_repo / "data" / "performance_log.csv").read_text() == "real perf\n"
