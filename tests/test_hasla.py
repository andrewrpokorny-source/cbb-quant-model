import pandas as pd
import pytest

from hasla import (
    HASLA_GAME_FEATURE_COLUMNS,
    add_hasla_features,
    build_team_map,
    ensure_hasla_feature_columns,
    matchup_features_for_game,
    parse_hasla_season_xml,
)


SAMPLE_XML = """
<mydata>
  <mr rk="1" t="Michigan" />
  <mr rk="2" t="Wisconsin" />
  <trd data="11/4,,,,,,,11/11" />
  <tr t="Michigan" data="10,9,8,7,6,5,4,3" />
  <tr t="Wisconsin" data="40,39,38,37,36,35,34,33" />
  <otr t="Michigan" data="11,10,9,8,7,6,5,4" />
  <otr t="Wisconsin" data="35,34,33,32,31,30,29,28" />
  <dtr t="Michigan" data="20,19,18,17,16,15,14,13" />
  <dtr t="Wisconsin" data="50,49,48,47,46,45,44,43" />
</mydata>
"""


def test_parse_hasla_season_xml_expands_daily_rank_history():
    result = parse_hasla_season_xml(SAMPLE_XML, season=2026)

    michigan = result[result["team"] == "Michigan"].reset_index(drop=True)
    assert len(michigan) == 8
    assert michigan.loc[0, "snapshot_date"] == "2025-11-04"
    assert michigan.loc[7, "snapshot_date"] == "2025-11-11"
    assert michigan.loc[0, "hasla_rank"] == 10
    assert michigan.loc[7, "hasla_off_rank"] == 4
    assert michigan.loc[7, "hasla_def_rank"] == 13


def test_build_team_map_matches_alias_variants():
    result = build_team_map(
        ["Abilene Christian Wildcats", "Middle Tennessee Blue Raiders"],
        ["Abil. Christian", "MTSU"],
    )
    lookup = result.set_index("team")["hasla_team"].to_dict()
    assert lookup["Abilene Christian Wildcats"] == "Abil. Christian"
    assert lookup["Middle Tennessee Blue Raiders"] == "MTSU"


def test_build_team_map_matches_common_hasla_aliases():
    result = build_team_map(
        [
            "Boston University Terriers",
            "Connecticut Huskies",
            "Saint Francis Red Flash",
            "Saint Joseph's Hawks",
            "SIU Edwardsville Cougars",
            "UMKC Kangaroos",
            "UC Santa Barbara Gauchos",
        ],
        ["Boston U", "UConn", "St. Francis (PA)", "Saint Joe's", "SIUE", "Kansas City", "UCSB"],
    )
    lookup = result.set_index("team")["hasla_team"].to_dict()
    assert lookup["Boston University Terriers"] == "Boston U"
    assert lookup["Connecticut Huskies"] == "UConn"
    assert lookup["Saint Francis Red Flash"] == "St. Francis (PA)"
    assert lookup["Saint Joseph's Hawks"] == "Saint Joe's"
    assert lookup["SIU Edwardsville Cougars"] == "SIUE"
    assert lookup["UMKC Kangaroos"] == "Kansas City"
    assert lookup["UC Santa Barbara Gauchos"] == "UCSB"


def test_add_hasla_features_uses_latest_snapshot_before_game_date():
    games = pd.DataFrame(
        [
            {"date": "2025-11-12", "team": "Michigan", "opponent": "Wisconsin"},
            {"date": "2025-11-12", "team": "Wisconsin", "opponent": "Michigan"},
        ]
    )
    snapshots = pd.DataFrame(
        [
            {"snapshot_date": "2025-11-11", "season": 2026, "team": "Michigan", "hasla_rank": 3, "hasla_off_rank": 4, "hasla_def_rank": 13},
            {"snapshot_date": "2025-11-11", "season": 2026, "team": "Wisconsin", "hasla_rank": 33, "hasla_off_rank": 28, "hasla_def_rank": 43},
            {"snapshot_date": "2025-11-12", "season": 2026, "team": "Michigan", "hasla_rank": 1, "hasla_off_rank": 1, "hasla_def_rank": 1},
        ]
    )
    team_map = pd.DataFrame(
        [
            {"team": "Michigan", "hasla_team": "Michigan", "match_source": "exact", "match_score": 1.0, "needs_review": False},
            {"team": "Wisconsin", "hasla_team": "Wisconsin", "match_source": "exact", "match_score": 1.0, "needs_review": False},
        ]
    )

    enriched = add_hasla_features(games, snapshots, team_map)
    home = enriched[enriched["team"] == "Michigan"].iloc[0]
    assert home["hasla_diff_rank_strength"] > 0
    assert home["hasla_diff_off_rank_strength"] > 0
    assert home["hasla_diff_def_rank_strength"] > 0


def test_matchup_features_for_game_defaults_to_zero_without_context():
    result = matchup_features_for_game("A", "B", "2026-02-15", pd.DataFrame(), pd.DataFrame())
    assert result == {col: 0.0 for col in HASLA_GAME_FEATURE_COLUMNS}


def test_ensure_hasla_feature_columns_adds_expected_defaults():
    df = ensure_hasla_feature_columns(pd.DataFrame({"team": ["Michigan"]}))
    for col in HASLA_GAME_FEATURE_COLUMNS:
        assert col in df.columns
        assert df.loc[0, col] == 0.0
