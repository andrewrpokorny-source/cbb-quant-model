import pandas as pd
import pytest

from torvik import (
    TORVIK_GAME_FEATURE_COLUMNS,
    add_torvik_features,
    build_missing_snapshot_dates,
    build_team_map,
    ensure_torvik_feature_columns,
    matchup_features_for_game,
    parse_ratings_html,
)


SAMPLE_HTML = """
<html>
  <body>
    <table>
      <caption>2026 T-Rank and Tempo-Free Stats</caption>
      <tr>
        <th>Rk</th><th>Team</th><th>Conf</th><th>G</th><th>Rec</th>
        <th>AdjOE</th><th>AdjDE</th><th>Barthag</th><th>EFG%</th><th>EFGD%</th>
        <th>TOR</th><th>TORD</th><th>ORB</th><th>DRB</th><th>FTR</th><th>FTRD</th>
        <th>2P%</th><th>2P%D</th><th>3P%</th><th>3P%D</th><th>3PR</th><th>3PRD</th>
        <th>Adj T.</th><th>WAB</th>
      </tr>
      <tr>
        <td>1</td><td>Michigan</td><td>B10</td><td>31</td><td>29-2</td>
        <td>129.1 4</td><td>91.4 3</td><td>.9816 1</td><td>58.7 9</td><td>44.2 1</td>
        <td>16.9 187</td><td>15.8 233</td><td>36.0 27</td><td>28.1 69</td><td>39.0 82</td><td>26.3 15</td>
        <td>61.6 2</td><td>44.1 3</td><td>36.5 37</td><td>29.6 7</td><td>42.2 131</td><td>41.7 265</td>
        <td>71.7 13</td><td>+12.9 1</td>
      </tr>
      <tr>
        <td>2</td><td>Wisconsin</td><td>B10</td><td>31</td><td>25-6</td>
        <td>121.0 22</td><td>95.0 10</td><td>.9420 8</td><td>54.0 31</td><td>47.5 35</td>
        <td>14.0 66</td><td>17.0 175</td><td>31.0 102</td><td>29.9 120</td><td>34.5 133</td><td>28.4 48</td>
        <td>55.2 40</td><td>46.2 22</td><td>34.9 110</td><td>31.2 68</td><td>40.1 175</td><td>36.8 102</td>
        <td>67.5 80</td><td>+8.1 10</td>
      </tr>
    </table>
  </body>
</html>
"""


def test_parse_ratings_html_extracts_selected_metrics():
    result = parse_ratings_html(
        SAMPLE_HTML,
        snapshot_date="2026-02-14",
        source_url="https://barttorvik.com/trank.php?year=2026",
    )

    assert list(result["team"]) == ["Michigan", "Wisconsin"]
    assert result.loc[0, "adj_oe"] == 129.1
    assert result.loc[0, "barthag"] == 0.9816
    assert result.loc[0, "adj_tempo"] == 71.7
    assert result.loc[0, "wab"] == 12.9


def test_parse_ratings_html_strips_embedded_tooltip_text_from_team_cell():
    html = SAMPLE_HTML.replace("<td>Michigan</td>", "<td>Michigan vs. 36 BYU (won)</td>")
    result = parse_ratings_html(
        html,
        snapshot_date="2026-02-14",
        source_url="https://barttorvik.com/trank.php?year=2026",
    )
    assert result.loc[0, "team"] == "Michigan"


def test_build_team_map_matches_normalized_names():
    result = build_team_map(
        team_names=["Saint Mary's", "NC State", "Michigan"],
        torvik_team_names=["St. Mary's", "N.C. State", "Michigan"],
    )

    lookup = result.set_index("team")["torvik_team"].to_dict()
    assert lookup["Saint Mary's"] == "St. Mary's"
    assert lookup["NC State"] == "N.C. State"
    assert lookup["Michigan"] == "Michigan"


def test_add_torvik_features_uses_latest_snapshot_before_game_date():
    games = pd.DataFrame(
        [
            {"date": "2026-02-15", "team": "Michigan", "opponent": "Wisconsin"},
            {"date": "2026-02-15", "team": "Wisconsin", "opponent": "Michigan"},
        ]
    )
    snapshots = pd.DataFrame(
        [
            {
                "snapshot_date": "2026-02-14",
                "season": 2026,
                "team": "Michigan",
                "conf": "B10",
                "games": 31,
                "record": "29-2",
                "adj_oe": 129.1,
                "adj_de": 91.4,
                "barthag": 0.9816,
                "adj_tempo": 71.7,
                "efg_off": 58.7,
                "efg_def": 44.2,
                "tor_off": 16.9,
                "tor_def": 15.8,
                "orb_off": 36.0,
                "drb_def": 28.1,
                "ftr_off": 39.0,
                "ftr_def": 26.3,
                "wab": 12.9,
                "source_url": "x",
            },
            {
                "snapshot_date": "2026-02-14",
                "season": 2026,
                "team": "Wisconsin",
                "conf": "B10",
                "games": 31,
                "record": "25-6",
                "adj_oe": 121.0,
                "adj_de": 95.0,
                "barthag": 0.9420,
                "adj_tempo": 67.5,
                "efg_off": 54.0,
                "efg_def": 47.5,
                "tor_off": 14.0,
                "tor_def": 17.0,
                "orb_off": 31.0,
                "drb_def": 29.9,
                "ftr_off": 34.5,
                "ftr_def": 28.4,
                "wab": 8.1,
                "source_url": "x",
            },
            {
                "snapshot_date": "2026-02-15",
                "season": 2026,
                "team": "Michigan",
                "conf": "B10",
                "games": 32,
                "record": "30-2",
                "adj_oe": 999.0,
                "adj_de": 999.0,
                "barthag": 0.9999,
                "adj_tempo": 99.0,
                "efg_off": 99.0,
                "efg_def": 99.0,
                "tor_off": 1.0,
                "tor_def": 1.0,
                "orb_off": 99.0,
                "drb_def": 99.0,
                "ftr_off": 99.0,
                "ftr_def": 99.0,
                "wab": 99.0,
                "source_url": "future",
            },
        ]
    )
    team_map = pd.DataFrame(
        [
            {"team": "Michigan", "torvik_team": "Michigan", "match_source": "exact", "match_score": 1.0, "needs_review": False},
            {"team": "Wisconsin", "torvik_team": "Wisconsin", "match_source": "exact", "match_score": 1.0, "needs_review": False},
        ]
    )

    enriched = add_torvik_features(games, snapshots, team_map)
    home = enriched[enriched["team"] == "Michigan"].iloc[0]

    assert home["torvik_team_adj_oe"] == 129.1
    assert home["torvik_opp_adj_oe"] == 121.0
    assert home["torvik_diff_adj_oe"] == pytest.approx(8.1)
    assert home["torvik_tempo_gap"] == pytest.approx(4.2)
    assert home["torvik_diff_ftr"] == pytest.approx(4.5)


def test_matchup_features_for_game_returns_zero_defaults_when_unmapped():
    snapshots = pd.DataFrame()
    team_map = pd.DataFrame()
    result = matchup_features_for_game("A", "B", "2026-02-15", snapshots, team_map)

    assert result == {col: 0.0 for col in TORVIK_GAME_FEATURE_COLUMNS}


def test_ensure_torvik_feature_columns_adds_all_expected_fields():
    df = ensure_torvik_feature_columns(pd.DataFrame({"team": ["Michigan"]}))
    for col in TORVIK_GAME_FEATURE_COLUMNS:
        assert col in df.columns
        assert df.loc[0, col] == 0.0


def test_build_missing_snapshot_dates_uses_previous_day():
    games = pd.DataFrame({"date": ["2026-02-15", "2026-02-16", "2026-02-16"]})
    result = build_missing_snapshot_dates(games)
    assert result == [pd.Timestamp("2026-02-14"), pd.Timestamp("2026-02-15")]
