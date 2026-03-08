"""Shared league configuration for men's and women's CBB workflows."""

import os


LEAGUE_ALIASES = {
    "mens": "mens",
    "men": "mens",
    "m": "mens",
    "cbb": "mens",
    "womens": "womens",
    "women": "womens",
    "w": "womens",
    "wcbb": "womens",
}


LEAGUE_SETTINGS = {
    "mens": {
        "label": "Men's CBB",
        "sport_path": "mens-college-basketball",
        "data_file": "cbb_training_data_processed.csv",
        "model_file": "cbb_model_v2.pkl",
        "win_model_file": "cbb_win_model_v1.pkl",
        "predictions_file": "daily_predictions.csv",
        "predictions_archive_prefix": "predictions",
        "performance_file": "performance_log.csv",
    },
    "womens": {
        "label": "Women's CBB",
        "sport_path": "womens-college-basketball",
        "data_file": "wbb_training_data_processed.csv",
        "model_file": "womens_cbb_spread_model_v2.pkl",
        "win_model_file": "womens_cbb_win_model_v1.pkl",
        "predictions_file": "daily_predictions_wbb.csv",
        "predictions_archive_prefix": "predictions_wbb",
        "performance_file": "performance_log_wbb.csv",
    },
}


def normalize_league(league):
    """Normalize league aliases to canonical keys."""
    normalized = LEAGUE_ALIASES.get(str(league or "").strip().lower())
    if normalized is None:
        raise ValueError(f"Unsupported league '{league}'. Use 'mens' or 'womens'.")
    return normalized


def get_league_settings(league):
    """Return canonical settings dict for the target league."""
    return LEAGUE_SETTINGS[normalize_league(league)]


def get_league_artifact_paths(base_dir, league):
    """Return absolute artifact paths for the target league."""
    settings = get_league_settings(league)
    return {
        "data_file": os.path.join(base_dir, settings["data_file"]),
        "model_file": os.path.join(base_dir, settings["model_file"]),
        "win_model_file": os.path.join(base_dir, settings["win_model_file"]),
        "odds_archive_file": os.path.join(base_dir, "odds_history.csv"),
        "predictions_file": os.path.join(base_dir, settings["predictions_file"]),
        "performance_file": os.path.join(base_dir, settings["performance_file"]),
        "predictions_archive_prefix": settings["predictions_archive_prefix"],
    }


def get_scoreboard_base_url(league):
    """Return ESPN scoreboard URL root for the target league."""
    settings = get_league_settings(league)
    return (
        "http://site.api.espn.com/apis/site/v2/sports/basketball/"
        f"{settings['sport_path']}/scoreboard?groups=50&limit=1000"
    )
