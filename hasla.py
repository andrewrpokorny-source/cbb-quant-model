import argparse
import json
import os
import re
from datetime import timedelta
from difflib import SequenceMatcher, get_close_matches
from typing import Iterable
from xml.etree import ElementTree as ET

import brotli
import pandas as pd
import requests

from league_config import get_league_artifact_paths, get_season_start_date, normalize_league


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEAM_MAP_FILE = os.path.join(BASE_DIR, "team_map.json")
HASLA_BASE_URL = "https://www.haslametrics.com/"
HASLA_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.haslametrics.com/",
}
HASLA_SNAPSHOT_COLUMNS = [
    "snapshot_date",
    "season",
    "team",
    "hasla_rank",
    "hasla_off_rank",
    "hasla_def_rank",
]
HASLA_GAME_FEATURE_COLUMNS = [
    "hasla_diff_rank_strength",
    "hasla_diff_off_rank_strength",
    "hasla_diff_def_rank_strength",
]
HASLA_MAP_COLUMNS = ["team", "hasla_team", "match_source", "match_score", "needs_review"]
MIN_EXPECTED_D1_TEAMS = 365.0
HASLA_NAME_OVERRIDES = {
    "abilene christian": "abil christian",
    "appalachian st": "app state",
    "arkansas pine bluff": "ar pine bluff",
    "boston university": "boston u",
    "cal st bakersfield": "csu bakersfield",
    "cal st fullerton": "csu fullerton",
    "cal st northridge": "csu northridge",
    "cal state bakersfield": "csu bakersfield",
    "cal state northridge": "csu northridge",
    "cal state fullerton": "csu fullerton",
    "central arkansas": "cent arkansas",
    "central connecticut": "cent conn st",
    "central connecticut st": "cent conn st",
    "central michigan": "cent michigan",
    "college of charleston": "charleston",
    "connecticut": "uconn",
    "east carolina": "ecu",
    "east tennessee st": "etsu",
    "east tennessee state": "etsu",
    "eastern illinois": "e illinois",
    "eastern kentucky": "e kentucky",
    "eastern michigan": "e michigan",
    "eastern washington": "e washington",
    "fairleigh dickinson": "fair dickinson",
    "florida atlantic": "fau",
    "florida gulf coast": "fgcu",
    "george washington": "g washington",
    "grambling": "grambling st",
    "grambling state": "grambling st",
    "grambling tigers": "grambling st",
    "iu indy": "iu indianapolis",
    "james madison": "jmu",
    "liu": "long island",
    "long island university": "liu",
    "loyola chicago": "loyola chicago",
    "loyola maryland": "loyola md",
    "maryland eastern shore": "umes",
    "massachusetts": "umass",
    "middle tennessee": "mtsu",
    "mississippi": "ole miss",
    "mississippi valley st": "miss valley st",
    "mississippi valley state": "miss valley st",
    "mount st marys": "mt st marys",
    "nc state": "nc state",
    "new hampshire": "unh",
    "north carolina greensboro": "uncg",
    "north carolina state": "nc state",
    "northern arizona": "n arizona",
    "northern colorado": "n colorado",
    "northern illinois": "n illinois",
    "northern kentucky": "n kentucky",
    "prairie view a and m": "pv a and m",
    "prairie view a&m": "pv a and m",
    "purdue fort wayne": "fort wayne",
    "rhode island": "uri",
    "saint bonaventure": "st bonaventure",
    "saint francis": "st francis pa",
    "saint johns": "st johns",
    "saint joseph's": "st joes",
    "saint josephs": "st joes",
    "st francis": "st francis pa",
    "st josephs": "st joes",
    "saint marys": "st marys",
    "saint peters": "st peters",
    "siu edwardsville": "siue",
    "south carolina state": "s carolina st",
    "south florida": "usf",
    "southeast missouri st": "se missouri st",
    "southeast missouri state": "se missouri st",
    "southeastern louisiana": "se louisiana",
    "southern indiana": "s indiana",
    "tennessee tech": "tenn tech",
    "tennessee martin": "ut martin",
    "texas a and m corpus christi": "texas a and m cc",
    "texas am corpus christi": "texas a&m corpus christi",
    "texas arlington": "ut arlington",
    "uc santa barbara": "ucsb",
    "ucf": "ucf",
    "uc san diego": "uc san diego",
    "unc greensboro": "uncg",
    "ul monroe": "ul monroe",
    "umkc": "kansas city",
    "st thomas minnesota": "st thomas",
    "ut rio grande valley": "utrgv",
    "western carolina": "w carolina",
    "western illinois": "w illinois",
    "western kentucky": "w kentucky",
    "western michigan": "w michigan",
}
_EXTERNAL_TEAM_MAP = None


class HaslaParseError(ValueError):
    """Raised when Haslametrics XML cannot be parsed into the expected schema."""


def _empty_snapshot_frame():
    return pd.DataFrame(columns=HASLA_SNAPSHOT_COLUMNS)


def _empty_match_frame(team_names: Iterable[str]):
    return pd.DataFrame(
        [
            {
                "team": team,
                "hasla_team": pd.NA,
                "match_source": "unmatched",
                "match_score": 0.0,
                "needs_review": True,
            }
            for team in sorted(set(team_names))
        ],
        columns=HASLA_MAP_COLUMNS,
    )


def _season_year_for_date(value):
    ts = pd.Timestamp(value).normalize()
    return ts.year + 1 if ts.month >= 11 else ts.year


def _season_start_for_year(season):
    return pd.Timestamp(year=season - 1, month=11, day=1)


def _season_file_name(season):
    suffix = "" if season == _season_year_for_date(pd.Timestamp.today()) else str(season - 2000)
    return f"ratings{suffix}.xml"


def _load_external_team_map():
    global _EXTERNAL_TEAM_MAP
    if _EXTERNAL_TEAM_MAP is not None:
        return _EXTERNAL_TEAM_MAP
    if os.path.exists(TEAM_MAP_FILE):
        with open(TEAM_MAP_FILE, "r") as fh:
            _EXTERNAL_TEAM_MAP = json.load(fh)
    else:
        _EXTERNAL_TEAM_MAP = {}
    return _EXTERNAL_TEAM_MAP


def _normalize_name(value):
    text = str(value or "").strip().lower()
    if not text:
        return ""
    text = text.replace("&amp;", "&")
    replacements = {
        "&": " and ",
        ".": " ",
        "'": "",
        "-": " ",
        "/": " ",
        "(": " ",
        ")": " ",
    }
    for src, dest in replacements.items():
        text = text.replace(src, dest)
    text = re.sub(r"\bsaint\b", "st", text)
    text = re.sub(r"\bstate\b", "st", text)
    text = re.sub(r"\bwest\b", "w", text)
    text = re.sub(r"\bgreat\s+washington\b", "g washington", text)
    text = re.sub(r"\bmiami ohio\b", "miami oh", text)
    text = re.sub(r"\s+", " ", text).strip()
    return HASLA_NAME_OVERRIDES.get(text, text)


def _candidate_match_keys(team):
    team = str(team or "").strip()
    if not team:
        return []
    candidates = [team]
    mapped = _load_external_team_map().get(team)
    if mapped:
        candidates.append(mapped)
    tokens = team.split()
    for trim in (1, 2):
        if len(tokens) > trim:
            candidates.append(" ".join(tokens[:-trim]))
    normalized = []
    for candidate in candidates:
        norm = _normalize_name(candidate)
        if norm and norm not in normalized:
            normalized.append(norm)
    return normalized


def _decode_response_content(response):
    content = response.content
    if content.lstrip().startswith(b"<?xml"):
        return content.decode("utf-8")

    encoding = response.headers.get("content-encoding", "").lower()
    if encoding == "br":
        content = brotli.decompress(content)
        return content.decode("utf-8")
    return content.decode("utf-8")


def _response_is_access_denied(text):
    lowered = str(text or "").lower()
    return "<title>access denied</title>" in lowered


def fetch_hasla_season_xml(season, session=None, timeout=20):
    own_session = session is None
    session = session or requests.Session()
    url = f"{HASLA_BASE_URL}{_season_file_name(season)}"
    try:
        response = session.get(url, headers=HASLA_HEADERS, timeout=timeout)
        response.raise_for_status()
        text = _decode_response_content(response)
        if _response_is_access_denied(text):
            raise HaslaParseError("Haslametrics denied access to the season XML.")
        return text
    finally:
        if own_session:
            session.close()


def _extract_history_dates(root, season):
    trd = root.find("trd")
    if trd is None:
        raise HaslaParseError("Haslametrics XML missing trd history dates.")

    tokens = (trd.attrib.get("data") or "").split(",")
    if not tokens:
        raise HaslaParseError("Haslametrics history date vector is empty.")

    first_label = next((token for token in tokens if token), None)
    if not first_label:
        raise HaslaParseError("Haslametrics history date vector has no anchor labels.")

    month, day = [int(part) for part in first_label.split("/", 1)]
    year = season - 1 if month >= 11 else season
    start_date = pd.Timestamp(year=year, month=month, day=day)
    return [start_date + timedelta(days=idx) for idx in range(len(tokens))]


def _parse_rank_series(element, dates, attr_name):
    team = element.attrib.get("t")
    values = (element.attrib.get("data") or "").split(",")
    if len(values) > len(dates):
        raise HaslaParseError(f"Haslametrics rank history longer than date vector for {team} ({attr_name}).")
    return {
        "team": team,
        attr_name: [int(value) if value else pd.NA for value in values],
    }


def parse_hasla_season_xml(xml_text, season):
    root = ET.fromstring(xml_text)
    dates = _extract_history_dates(root, season)

    all_play = {}
    for element in root.findall("tr"):
        parsed = _parse_rank_series(element, dates, "hasla_rank")
        all_play[parsed["team"]] = parsed["hasla_rank"]

    off_rank = {}
    for element in root.findall("otr"):
        parsed = _parse_rank_series(element, dates, "hasla_off_rank")
        off_rank[parsed["team"]] = parsed["hasla_off_rank"]

    def_rank = {}
    for element in root.findall("dtr"):
        parsed = _parse_rank_series(element, dates, "hasla_def_rank")
        def_rank[parsed["team"]] = parsed["hasla_def_rank"]

    teams = sorted(set(all_play) & set(off_rank) & set(def_rank))
    rows = []
    for team in teams:
        series_len = min(len(all_play[team]), len(off_rank[team]), len(def_rank[team]), len(dates))
        for idx, date in enumerate(dates[:series_len]):
            rows.append(
                {
                    "snapshot_date": date.strftime("%Y-%m-%d"),
                    "season": season,
                    "team": team,
                    "hasla_rank": all_play[team][idx],
                    "hasla_off_rank": off_rank[team][idx],
                    "hasla_def_rank": def_rank[team][idx],
                }
            )

    return pd.DataFrame(rows, columns=HASLA_SNAPSHOT_COLUMNS)


def load_snapshot_file(path):
    if not path or not os.path.exists(path):
        return _empty_snapshot_frame()
    df = pd.read_csv(path)
    if df.empty:
        return _empty_snapshot_frame()
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"]).dt.strftime("%Y-%m-%d")
    return df[HASLA_SNAPSHOT_COLUMNS].copy()


def save_snapshot_file(df, path):
    ordered = df.copy()
    for col in HASLA_SNAPSHOT_COLUMNS:
        if col not in ordered.columns:
            ordered[col] = pd.NA
    ordered = ordered[HASLA_SNAPSHOT_COLUMNS].sort_values(["snapshot_date", "team"]).reset_index(drop=True)
    ordered.to_csv(path, index=False)


def load_team_map(path, team_names=None):
    if path and os.path.exists(path):
        df = pd.read_csv(path)
        for col in HASLA_MAP_COLUMNS:
            if col not in df.columns:
                df[col] = pd.NA
        return df[HASLA_MAP_COLUMNS].copy()
    return _empty_match_frame(list(team_names) if team_names is not None else [])


def save_team_map(df, path):
    ordered = df.copy()
    for col in HASLA_MAP_COLUMNS:
        if col not in ordered.columns:
            ordered[col] = pd.NA
    ordered = ordered[HASLA_MAP_COLUMNS].sort_values("team").reset_index(drop=True)
    ordered.to_csv(path, index=False)


def build_team_map(team_names, hasla_team_names, existing_map=None):
    existing_map = existing_map if existing_map is not None else _empty_match_frame([])
    existing_lookup = existing_map.set_index("team").to_dict("index") if not existing_map.empty else {}
    normalized_lookup = {}
    for hasla_team in sorted(set(str(name) for name in hasla_team_names if str(name).strip())):
        normalized_lookup.setdefault(_normalize_name(hasla_team), []).append(hasla_team)

    rows = []
    for team in sorted(set(str(name) for name in team_names if str(name).strip())):
        existing_row = existing_lookup.get(team)
        if existing_row and pd.notna(existing_row.get("hasla_team")):
            rows.append(
                {
                    "team": team,
                    "hasla_team": existing_row["hasla_team"],
                    "match_source": existing_row.get("match_source", "manual"),
                    "match_score": float(existing_row.get("match_score", 1.0) or 1.0),
                    "needs_review": bool(existing_row.get("needs_review", False)),
                }
            )
            continue

        candidate_keys = _candidate_match_keys(team)
        matched = None
        for key in candidate_keys:
            candidates = normalized_lookup.get(key, [])
            if len(candidates) == 1:
                matched = (candidates[0], key, 1.0, "normalized_exact" if key == _normalize_name(team) else "alias_exact")
                break

        if matched is None:
            best = None
            for key in candidate_keys:
                close = get_close_matches(key, list(normalized_lookup), n=2, cutoff=0.82)
                if not close:
                    continue
                best_key = close[0]
                best_score = SequenceMatcher(None, key, best_key).ratio()
                second_score = SequenceMatcher(None, key, close[1]).ratio() if len(close) > 1 else 0.0
                candidates = normalized_lookup.get(best_key, [])
                if len(candidates) != 1:
                    continue
                if best is None or best_score > best[2]:
                    best = (candidates[0], key, best_score, second_score)
            if best is not None and best[2] >= 0.9 and (best[2] - best[3]) >= 0.03:
                matched = (best[0], best[1], best[2], "fuzzy" if best[1] == _normalize_name(team) else "alias_fuzzy")

        if matched is not None:
            rows.append(
                {
                    "team": team,
                    "hasla_team": matched[0],
                    "match_source": matched[3],
                    "match_score": round(float(matched[2]), 4),
                    "needs_review": False,
                }
            )
            continue

        rows.append(
            {
                "team": team,
                "hasla_team": pd.NA,
                "match_source": "unmatched",
                "match_score": 0.0,
                "needs_review": True,
            }
        )

    return pd.DataFrame(rows, columns=HASLA_MAP_COLUMNS)


def sync_team_map_for_games(games_df, snapshots_df, team_map_df=None):
    all_teams = pd.concat([games_df["team"], games_df["opponent"]], ignore_index=True).dropna().unique()
    snapshot_teams = snapshots_df["team"].dropna().unique() if not snapshots_df.empty else []
    return build_team_map(all_teams, snapshot_teams, existing_map=team_map_df)


def ensure_hasla_feature_columns(df):
    for col in HASLA_GAME_FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0
    return df


def _merge_side_snapshot(df, snapshots_df, left_team_col, prefix):
    left = df.copy()
    left[left_team_col] = left[left_team_col].fillna("")
    left["lookup_date"] = pd.to_datetime(left["date"]) - timedelta(days=1)
    left = left.sort_values(["lookup_date", left_team_col]).reset_index(drop=True)

    right = snapshots_df.copy()
    right["snapshot_date"] = pd.to_datetime(right["snapshot_date"])
    right = right.rename(columns={"team": left_team_col})
    right[left_team_col] = right[left_team_col].fillna("")
    right = right.sort_values(["snapshot_date", left_team_col]).reset_index(drop=True)

    merged = pd.merge_asof(
        left,
        right,
        left_on="lookup_date",
        right_on="snapshot_date",
        by=left_team_col,
        direction="backward",
        allow_exact_matches=True,
    )

    rename_map = {
        "hasla_rank": f"{prefix}hasla_rank",
        "hasla_off_rank": f"{prefix}hasla_off_rank",
        "hasla_def_rank": f"{prefix}hasla_def_rank",
    }
    merged = merged.rename(columns=rename_map)
    drop_cols = [col for col in ["snapshot_date", "season", "lookup_date"] if col in merged.columns]
    return merged.drop(columns=drop_cols)


def _rank_to_strength(series):
    numeric = pd.to_numeric(series, errors="coerce")
    max_rank = numeric.max(skipna=True)
    scale_max = max(float(max_rank) if pd.notna(max_rank) else MIN_EXPECTED_D1_TEAMS, MIN_EXPECTED_D1_TEAMS)
    clipped = numeric.clip(lower=1.0, upper=scale_max)
    return 1.0 - ((clipped - 1.0) / max(scale_max - 1.0, 1.0))


def add_hasla_features(df, snapshots_df, team_map_df):
    df = ensure_hasla_feature_columns(df.copy())
    if df.empty or snapshots_df.empty or team_map_df.empty:
        return df

    mapped = df.drop(columns=[col for col in HASLA_GAME_FEATURE_COLUMNS if col in df.columns]).copy()
    lookup = team_map_df[["team", "hasla_team"]].drop_duplicates()
    mapped = mapped.merge(lookup.rename(columns={"team": "team", "hasla_team": "team_hasla"}), on="team", how="left")
    mapped = mapped.merge(
        lookup.rename(columns={"team": "opponent", "hasla_team": "opponent_hasla"}),
        on="opponent",
        how="left",
    )

    snapshots = snapshots_df.copy()
    merged = _merge_side_snapshot(mapped, snapshots, "team_hasla", "team_")
    merged = _merge_side_snapshot(merged, snapshots, "opponent_hasla", "opp_")

    merged["hasla_diff_rank_strength"] = _rank_to_strength(merged.get("team_hasla_rank")) - _rank_to_strength(
        merged.get("opp_hasla_rank")
    )
    merged["hasla_diff_off_rank_strength"] = _rank_to_strength(merged.get("team_hasla_off_rank")) - _rank_to_strength(
        merged.get("opp_hasla_off_rank")
    )
    merged["hasla_diff_def_rank_strength"] = _rank_to_strength(merged.get("team_hasla_def_rank")) - _rank_to_strength(
        merged.get("opp_hasla_def_rank")
    )

    for col in HASLA_GAME_FEATURE_COLUMNS:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)

    drop_cols = [
        "team_hasla",
        "opponent_hasla",
        "team_hasla_rank",
        "team_hasla_off_rank",
        "team_hasla_def_rank",
        "opp_hasla_rank",
        "opp_hasla_off_rank",
        "opp_hasla_def_rank",
    ]
    merged = merged.drop(columns=[col for col in drop_cols if col in merged.columns])
    return ensure_hasla_feature_columns(merged)


def matchup_features_for_game(home_team, away_team, game_date, snapshots_df, team_map_df):
    frame = pd.DataFrame([{"team": home_team, "opponent": away_team, "date": pd.Timestamp(game_date).strftime("%Y-%m-%d")}])
    enriched = add_hasla_features(frame, snapshots_df, team_map_df)
    if enriched.empty:
        return {col: 0.0 for col in HASLA_GAME_FEATURE_COLUMNS}
    row = enriched.iloc[0]
    return {col: float(pd.to_numeric(row.get(col), errors="coerce") or 0.0) for col in HASLA_GAME_FEATURE_COLUMNS}


def build_required_seasons(games_df, league):
    if games_df.empty or "date" not in games_df.columns:
        return []
    season_start = pd.Timestamp(get_season_start_date(league))
    dates = pd.to_datetime(games_df["date"], errors="coerce")
    dates = dates[dates >= season_start]
    return sorted({_season_year_for_date(date) for date in dates.dropna()})


def sync_from_processed_data(league="mens", timeout=20):
    league = normalize_league(league)
    if league != "mens":
        raise ValueError("Haslametrics sync is currently supported for the men's pipeline only.")

    paths = get_league_artifact_paths(BASE_DIR, league)
    data_file = paths["data_file"]
    snapshot_file = paths["hasla_snapshot_file"]
    map_file = paths["hasla_map_file"]
    if not data_file or not os.path.exists(data_file):
        raise FileNotFoundError(f"Processed data file not found: {data_file}")

    games_df = pd.read_csv(data_file, low_memory=False)
    seasons = build_required_seasons(games_df, league)
    combined = load_snapshot_file(snapshot_file)
    existing_seasons = set(combined["season"].dropna().astype(int)) if not combined.empty else set()

    session = requests.Session()
    try:
        for season in seasons:
            if season in existing_seasons:
                continue
            print(f"   -> Haslametrics season {season}")
            xml_text = fetch_hasla_season_xml(season, session=session, timeout=timeout)
            fetched = parse_hasla_season_xml(xml_text, season)
            combined = pd.concat([combined, fetched], ignore_index=True)
            combined["snapshot_date"] = pd.to_datetime(combined["snapshot_date"]).dt.strftime("%Y-%m-%d")
            combined = combined.drop_duplicates(subset=["snapshot_date", "team"], keep="last")
            save_snapshot_file(combined, snapshot_file)
    finally:
        session.close()

    snapshots = load_snapshot_file(snapshot_file)
    team_map = load_team_map(map_file, pd.concat([games_df["team"], games_df["opponent"]]).dropna().unique())
    team_map = sync_team_map_for_games(games_df, snapshots, team_map_df=team_map)
    save_team_map(team_map, map_file)

    mapped_teams = team_map.loc[team_map["hasla_team"].notna(), "team"].nunique() if not team_map.empty else 0
    all_teams = pd.concat([games_df["team"], games_df["opponent"]], ignore_index=True).dropna().nunique()
    return {
        "games": int(len(games_df)),
        "team_map_coverage": (mapped_teams / all_teams) if all_teams else 0.0,
        "snapshot_dates": int(pd.to_datetime(snapshots["snapshot_date"], errors="coerce").dropna().nunique())
        if not snapshots.empty
        else 0,
        "seasons": seasons,
    }


def coverage_report(league="mens"):
    league = normalize_league(league)
    paths = get_league_artifact_paths(BASE_DIR, league)
    games_df = pd.read_csv(paths["data_file"], low_memory=False) if paths["data_file"] and os.path.exists(paths["data_file"]) else pd.DataFrame()
    snapshots = load_snapshot_file(paths["hasla_snapshot_file"])
    team_map = load_team_map(paths["hasla_map_file"], pd.concat([games_df.get("team", pd.Series(dtype=str)), games_df.get("opponent", pd.Series(dtype=str))]).dropna().unique())
    mapped_teams = team_map.loc[team_map["hasla_team"].notna(), "team"].nunique() if not team_map.empty else 0
    all_teams = pd.concat([games_df.get("team", pd.Series(dtype=str)), games_df.get("opponent", pd.Series(dtype=str))], ignore_index=True).dropna().nunique()
    return {
        "games": int(len(games_df)),
        "team_map_coverage": (mapped_teams / all_teams) if all_teams else 0.0,
        "snapshot_dates": int(pd.to_datetime(snapshots["snapshot_date"], errors="coerce").dropna().nunique())
        if not snapshots.empty
        else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Manage historical Haslametrics rank snapshots.")
    parser.add_argument("command", choices=["sync", "coverage"], help="sync or coverage report")
    parser.add_argument("--league", default="mens", help="League to operate on")
    parser.add_argument("--timeout", type=int, default=20, help="HTTP timeout in seconds")
    args = parser.parse_args()

    if args.command == "sync":
        report = sync_from_processed_data(league=args.league, timeout=args.timeout)
        if report.get("seasons"):
            print("seasons=" + ",".join(str(season) for season in report["seasons"]))
    else:
        report = coverage_report(league=args.league)

    print(f"games={report['games']}")
    print(f"team_map_coverage={report['team_map_coverage']:.1%}")
    print(f"snapshot_dates={report['snapshot_dates']}")


if __name__ == "__main__":
    main()
