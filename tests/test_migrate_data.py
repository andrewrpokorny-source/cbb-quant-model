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


def test_migrate_moves_prediction_archives(fake_repo):
    """Glob-based migration moves prediction archive CSVs to data/."""
    from migrate_data import migrate

    (fake_repo / "predictions_20260323.csv").write_text("mens archive\n")
    (fake_repo / "predictions_wbb_20260110.csv").write_text("wbb archive\n")
    (fake_repo / "predictions_mlb_20260401.csv").write_text("mlb archive\n")

    with patch("migrate_data.BASE_DIR", str(fake_repo)):
        result = migrate()

    assert result is True
    assert not (fake_repo / "predictions_20260323.csv").exists()
    assert (fake_repo / "data" / "predictions_20260323.csv").read_text() == "mens archive\n"
    assert not (fake_repo / "predictions_wbb_20260110.csv").exists()
    assert (fake_repo / "data" / "predictions_wbb_20260110.csv").read_text() == "wbb archive\n"
    assert not (fake_repo / "predictions_mlb_20260401.csv").exists()
    assert (fake_repo / "data" / "predictions_mlb_20260401.csv").read_text() == "mlb archive\n"


def test_migrate_moves_pkl_to_models(fake_repo):
    """PKL files should migrate to models/, not data/."""
    from migrate_data import migrate

    (fake_repo / "wbb_model_v2.pkl").write_bytes(b"fake model")

    with patch("migrate_data.BASE_DIR", str(fake_repo)):
        result = migrate()

    assert result is True
    assert not (fake_repo / "wbb_model_v2.pkl").exists()
    assert (fake_repo / "models" / "wbb_model_v2.pkl").read_bytes() == b"fake model"


def test_migrate_idempotent_on_clean_repo(fake_repo):
    """Running migrate on an already-migrated repo does nothing and returns True."""
    from migrate_data import migrate

    with patch("migrate_data.BASE_DIR", str(fake_repo)):
        result = migrate()

    assert result is True


def test_migrate_partial_success_with_conflict(fake_repo):
    """Non-conflicting files still move even when one file has a conflict."""
    from migrate_data import migrate

    # This one will conflict
    (fake_repo / "betting_history.csv").write_text("old ledger\n")
    (fake_repo / "data" / "betting_history.csv").write_text("new ledger\n")
    # This one should move fine
    (fake_repo / "performance_log.csv").write_text("perf data\n")

    with patch("migrate_data.BASE_DIR", str(fake_repo)):
        result = migrate()

    assert result is False  # conflict exists
    # Conflict untouched
    assert (fake_repo / "betting_history.csv").exists()
    assert (fake_repo / "data" / "betting_history.csv").read_text() == "new ledger\n"
    # Non-conflicting file still moved
    assert not (fake_repo / "performance_log.csv").exists()
    assert (fake_repo / "data" / "performance_log.csv").read_text() == "perf data\n"


def test_migrate_creates_directories(tmp_path):
    """migrate() should create data/ and models/ if they don't exist."""
    from migrate_data import migrate

    (tmp_path / "betting_history.csv").write_text("ledger\n")

    with patch("migrate_data.BASE_DIR", str(tmp_path)):
        result = migrate()

    assert result is True
    assert (tmp_path / "data").is_dir()
    assert (tmp_path / "models").is_dir()
    assert (tmp_path / "data" / "betting_history.csv").read_text() == "ledger\n"
