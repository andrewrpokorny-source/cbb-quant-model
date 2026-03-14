import pandas as pd
import pytest

from womens_net import (
    WOMENS_NET_GAME_FEATURE_COLUMNS,
    add_womens_net_features,
    build_team_map,
    ensure_womens_net_feature_columns,
    load_snapshot_file,
    parse_current_net_html,
)


SAMPLE_HTML = """
<html>
  <body>
    <div>Through Games Mar. 12 2026</div>
    <table class="sticky">
      <tr>
        <th>Rank</th><th>School</th><th>Record</th><th>Conf</th><th>Road</th><th>Neutral</th>
        <th>Home</th><th>Non-Div I</th><th>Prev</th><th>Quad 1</th><th>Quad 2</th><th>Quad 3</th><th>Quad 4</th>
      </tr>
      <tr>
        <td>1</td><td>UConn</td><td>34-0</td><td>Big East</td><td>12-0</td><td>7-0</td>
        <td>15-0</td><td>0-0</td><td>1</td><td>9-0</td><td>5-0</td><td>8-0</td><td>12-0</td>
      </tr>
      <tr>
        <td>6</td><td>Michigan</td><td>25-6</td><td>Big Ten</td><td>8-2</td><td>3-3</td>
        <td>14-1</td><td>0-0</td><td>6</td><td>9-6</td><td>4-0</td><td>4-0</td><td>8-0</td>
      </tr>
    </table>
  </body>
</html>
"""


def test_parse_current_net_html_extracts_rank_rows():
    result = parse_current_net_html(SAMPLE_HTML)

    assert list(result["team"]) == ["UConn", "Michigan"]
    assert result.loc[0, "snapshot_date"] == "2026-03-12"
    assert result.loc[0, "net_rank"] == 1
    assert result.loc[1, "prev_rank"] == 6


def test_build_team_map_matches_common_aliases():
    result = build_team_map(
        ["Connecticut Huskies", "Michigan Wolverines"],
        ["UConn", "Michigan"],
    )
    lookup = result.set_index("team")["net_team"].to_dict()
    assert lookup["Connecticut Huskies"] == "UConn"
    assert lookup["Michigan Wolverines"] == "Michigan"


def test_build_team_map_avoids_false_positive_state_aliases():
    result = build_team_map(
        [
            "Florida International Panthers",
            "Florida Gators",
            "Texas A&M-International Dustdevils",
            "Texas Longhorns",
        ],
        ["FIU", "Florida", "Texas A&M", "Texas"],
    )
    lookup = result.set_index("team")["net_team"].to_dict()
    assert lookup["Florida International Panthers"] == "FIU"
    assert lookup["Florida Gators"] == "Florida"
    assert pd.isna(lookup["Texas A&M-International Dustdevils"])
    assert lookup["Texas Longhorns"] == "Texas"


def test_build_team_map_matches_common_abbreviated_net_names():
    result = build_team_map(
        [
            "Boston University Terriers",
            "Florida Gulf Coast Eagles",
            "Middle Tennessee Blue Raiders",
            "UNC Wilmington Seahawks",
        ],
        ["Boston U.", "FGCU", "Middle Tenn.", "UNCW"],
    )
    lookup = result.set_index("team")["net_team"].to_dict()
    assert lookup["Boston University Terriers"] == "Boston U."
    assert lookup["Florida Gulf Coast Eagles"] == "FGCU"
    assert lookup["Middle Tennessee Blue Raiders"] == "Middle Tenn."
    assert lookup["UNC Wilmington Seahawks"] == "UNCW"


def test_build_team_map_recomputes_stale_auto_matches():
    existing = pd.DataFrame(
        [
            {
                "team": "Florida International Panthers",
                "net_team": "Florida",
                "match_source": "alias_exact",
                "match_score": 1.0,
                "needs_review": False,
            }
        ]
    )
    result = build_team_map(
        ["Florida International Panthers"],
        ["FIU", "Florida"],
        existing_map=existing,
    )
    assert result.iloc[0]["net_team"] == "FIU"
    assert result.iloc[0]["match_source"] != "manual"


def test_add_womens_net_features_uses_latest_prior_snapshot():
    games = pd.DataFrame(
        [
            {"date": "2026-03-13", "team": "Connecticut Huskies", "opponent": "Michigan Wolverines"},
            {"date": "2026-03-13", "team": "Michigan Wolverines", "opponent": "Connecticut Huskies"},
        ]
    )
    snapshots = pd.DataFrame(
        [
            {"snapshot_date": "2026-03-12", "team": "UConn", "net_rank": 1, "prev_rank": 1, "record": "34-0", "conf": "Big East", "quad1": "9-0", "quad2": "5-0", "quad3": "8-0", "quad4": "12-0", "source_url": "x"},
            {"snapshot_date": "2026-03-12", "team": "Michigan", "net_rank": 6, "prev_rank": 6, "record": "25-6", "conf": "Big Ten", "quad1": "9-6", "quad2": "4-0", "quad3": "4-0", "quad4": "8-0", "source_url": "x"},
            {"snapshot_date": "2026-03-13", "team": "UConn", "net_rank": 9, "prev_rank": 9, "record": "34-1", "conf": "Big East", "quad1": "9-1", "quad2": "5-0", "quad3": "8-0", "quad4": "12-0", "source_url": "future"},
        ]
    )
    team_map = pd.DataFrame(
        [
            {"team": "Connecticut Huskies", "net_team": "UConn", "match_source": "alias_exact", "match_score": 1.0, "needs_review": False},
            {"team": "Michigan Wolverines", "net_team": "Michigan", "match_source": "alias_exact", "match_score": 1.0, "needs_review": False},
        ]
    )

    enriched = add_womens_net_features(games, snapshots, team_map)
    home = enriched[enriched["team"] == "Connecticut Huskies"].iloc[0]

    assert home["womens_net_diff_rank_strength"] > 0
    assert home["womens_net_diff_prev_rank_strength"] > 0


def test_ensure_womens_net_feature_columns_adds_defaults():
    df = ensure_womens_net_feature_columns(pd.DataFrame({"team": ["UConn"]}))
    for col in WOMENS_NET_GAME_FEATURE_COLUMNS:
        assert col in df.columns
        assert df.loc[0, col] == 0.0


def test_load_snapshot_file_backfills_missing_columns(tmp_path):
    path = tmp_path / "womens_net.csv"
    pd.DataFrame([{"snapshot_date": "2026-03-12", "team": "UConn", "net_rank": 1}]).to_csv(path, index=False)

    loaded = load_snapshot_file(path)

    assert "prev_rank" in loaded.columns
    assert pd.isna(loaded.loc[0, "prev_rank"])
