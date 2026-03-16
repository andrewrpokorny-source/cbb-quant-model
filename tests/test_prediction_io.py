import pandas as pd

from prediction_io import load_predictions_csv


def test_load_predictions_csv_returns_missing_for_absent_file(tmp_path):
    df, status = load_predictions_csv(str(tmp_path / "missing.csv"))

    assert status == "missing"
    assert df.empty


def test_load_predictions_csv_treats_whitespace_only_file_as_empty(tmp_path):
    pred_file = tmp_path / "daily_predictions.csv"
    pred_file.write_text("\n", encoding="utf-8")

    df, status = load_predictions_csv(str(pred_file))

    assert status == "empty"
    assert df.empty


def test_load_predictions_csv_preserves_header_only_csv(tmp_path):
    pred_file = tmp_path / "daily_predictions.csv"
    pred_file.write_text("Bet_Type,Conf\n", encoding="utf-8")

    df, status = load_predictions_csv(str(pred_file))

    assert status == "loaded"
    assert list(df.columns) == ["Bet_Type", "Conf"]
    assert df.empty
