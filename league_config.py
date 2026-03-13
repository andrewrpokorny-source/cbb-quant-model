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
        "season_start_date": "2025-11-04",
        "data_file": "cbb_training_data_processed.csv",
        "torvik_snapshot_file": "torvik_ratings_snapshots.csv",
        "torvik_map_file": "torvik_team_map.csv",
        "hasla_snapshot_file": "hasla_rank_snapshots.csv",
        "hasla_map_file": "hasla_team_map.csv",
        "model_file": "cbb_model_v2.pkl",
        "win_model_file": "cbb_win_model_v1.pkl",
        "predictions_file": "daily_predictions.csv",
        "predictions_archive_prefix": "predictions",
        "performance_file": "performance_log.csv",
    },
    "womens": {
        "label": "Women's CBB",
        "sport_path": "womens-college-basketball",
        "season_start_date": "2025-11-05",
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
    torvik_snapshot_file = settings.get("torvik_snapshot_file")
    torvik_map_file = settings.get("torvik_map_file")
    hasla_snapshot_file = settings.get("hasla_snapshot_file")
    hasla_map_file = settings.get("hasla_map_file")
    return {
        "data_file": os.path.join(base_dir, settings["data_file"]),
        "torvik_snapshot_file": (
            os.path.join(base_dir, torvik_snapshot_file) if torvik_snapshot_file else None
        ),
        "torvik_map_file": os.path.join(base_dir, torvik_map_file) if torvik_map_file else None,
        "hasla_snapshot_file": os.path.join(base_dir, hasla_snapshot_file) if hasla_snapshot_file else None,
        "hasla_map_file": os.path.join(base_dir, hasla_map_file) if hasla_map_file else None,
        "model_file": os.path.join(base_dir, settings["model_file"]),
        "win_model_file": os.path.join(base_dir, settings["win_model_file"]),
        "odds_archive_file": os.path.join(base_dir, "odds_history.csv"),
        "kalshi_game_archive_file": os.path.join(base_dir, "kalshi_game_history.csv"),
        "predictions_file": os.path.join(base_dir, settings["predictions_file"]),
        "performance_file": os.path.join(base_dir, settings["performance_file"]),
        "predictions_archive_prefix": settings["predictions_archive_prefix"],
    }


def get_scoreboard_base_url(league):
    """Return ESPN scoreboard URL root for the target league."""
    settings = get_league_settings(league)
    return (
        "https://site.api.espn.com/apis/site/v2/sports/basketball/"
        f"{settings['sport_path']}/scoreboard?groups=50&limit=1000"
    )


def get_season_start_date(league):
    """Return the configured in-season bootstrap date for the target league."""
    settings = get_league_settings(league)
    return settings["season_start_date"]
