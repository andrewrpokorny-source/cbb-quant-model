import argparse
import html
import json
import os
import re
from difflib import SequenceMatcher, get_close_matches

import pandas as pd
import requests

from league_config import get_league_artifact_paths, normalize_league


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEAM_MAP_FILE = os.path.join(BASE_DIR, "team_map.json")
WOMENS_NET_URL = "https://www.ncaa.com/rankings/basketball-women/d1/ncaa-womens-basketball-net-rankings"
WOMENS_NET_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.ncaa.com/",
}
WOMENS_NET_SNAPSHOT_COLUMNS = [
    "snapshot_date",
    "team",
    "net_rank",
    "prev_rank",
    "record",
    "conf",
    "quad1",
    "quad2",
    "quad3",
    "quad4",
    "source_url",
]
WOMENS_NET_GAME_FEATURE_COLUMNS = [
    "womens_net_diff_rank_strength",
    "womens_net_diff_prev_rank_strength",
]
WOMENS_NET_MAP_COLUMNS = ["team", "net_team", "match_source", "match_score", "needs_review"]
# A substantially smaller parsed field usually means the NCAA table layout changed.
MIN_WOMENS_NET_TEAMS = 360
WOMENS_NET_NAME_OVERRIDES = {
    "connecticut": "uconn",
    "mississippi": "ole miss",
    "saint josephs": "saint joseph's",
    "saint marys": "saint mary's",
    "saint johns": "st. john's",
    "southern cal": "usc",
    "southern california": "usc",
    "texas christian": "tcu",
    "southern methodist": "smu",
    "louisiana state": "lsu",
    "massachusetts": "umass",
    "alcorn state": "alcorn",
    "alcorn st": "alcorn",
    "arkansas pine bluff golden lions": "ark pine bluff",
    "arkansas pine bluff": "ark pine bluff",
    "army black": "army west point",
    "army black knights": "army west point",
    "boston university": "boston u",
    "central arkansas": "central ark",
    "central connecticut": "central conn st",
    "central connecticut blue": "central conn st",
    "central michigan": "central mich",
    "charleston southern": "charleston so",
    "east tennessee state": "etsu",
    "eastern illinois": "eastern ill",
    "eastern kentucky": "eastern ky",
    "eastern michigan": "eastern mich",
    "eastern washington": "eastern wash",
    "fairleigh dickinson": "fdu",
    "florida atlantic": "fla atlantic",
    "florida gulf coast": "fgcu",
    "florida international": "fiu",
    "georgia southern": "ga southern",
    "iu indianapolis": "iu indy",
    "incarnate word": "uiw",
    "lamar": "lamar u",
    "long island u": "liu",
    "long island university": "liu",
    "loyola marymount": "lmu ca",
    "maryland eastern shore": "umes",
    "middle tennessee": "middle tenn",
    "north alabama": "north ala",
    "north carolina a and t": "nc a and t",
    "north carolina central": "nc central",
    "northern arizona": "northern ariz",
    "northern colorado": "northern colo",
    "northern illinois": "niu",
    "northern iowa": "uni",
    "northern kentucky": "northern ky",
    "prairie view a and m": "prairie view",
    "queens university": "queens nc",
    "saint marys gaels": "saint mary's ca",
    "siu edwardsville": "siue",
    "south florida": "south fla",
    "southeast missouri state": "southeast mo st",
    "southeastern louisiana": "southeastern la",
    "southern illinois": "southern ill",
    "southern indiana": "southern ind",
    "southern university": "southern u",
    "st johns red storm": "st john's ny",
    "texas a and m corpus christi islanders": "a and m corpus christi",
    "ul monroe": "ulm",
    "unc wilmington": "uncw",
    "ut rio grande valley": "utrgv",
    "west georgia": "west ga",
    "western carolina": "western caro",
    "western illinois": "western ill",
    "western kentucky": "western ky",
    "western michigan": "western mich",
}
_EXTERNAL_TEAM_MAP = None


class WomensNetParseError(ValueError):
    """Raised when NCAA women's NET HTML cannot be parsed."""


def _empty_snapshot_frame():
    return pd.DataFrame(columns=WOMENS_NET_SNAPSHOT_COLUMNS)


def _empty_match_frame(team_names):
    return pd.DataFrame(
        [
            {
                "team": team,
                "net_team": pd.NA,
                "match_source": "unmatched",
                "match_score": 0.0,
                "needs_review": True,
            }
            for team in sorted(set(team_names))
        ],
        columns=WOMENS_NET_MAP_COLUMNS,
    )


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
    text = html.unescape(text)
    replacements = {
        "&amp;": " and ",
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
    text = re.sub(r"\buniversity\b", "u", text)
    text = re.sub(r"\bcollege\b", "col", text)
    text = re.sub(r"\s+", " ", text).strip()
    return WOMENS_NET_NAME_OVERRIDES.get(text, text)


def _candidate_match_keys(team):
    team = str(team or "").strip()
    if not team:
        return []
    candidates = [team]
    mapped = _load_external_team_map().get(team)
    if mapped:
        candidates.append(mapped)
    normalized = []
    for candidate in candidates:
        norm = _normalize_name(candidate)
        if norm and norm not in normalized:
            normalized.append(norm)
        tokens = norm.split()
        max_trim = min(2, len(tokens))
        if len(tokens) > 4:
            max_trim = 1
        for trim in range(1, max_trim + 1):
            base_shortened = " ".join(tokens[:-trim]).strip()
            shortened = _normalize_name(base_shortened)
            allow_shortened = (
                len(base_shortened.split()) >= 2
                or shortened != base_shortened
                or (len(tokens) == 2 and len(base_shortened.split()) == 1)
            )
            if shortened and allow_shortened and shortened not in normalized:
                normalized.append(shortened)
    return normalized


def _clean_cell(value):
    text = re.sub(r"<[^>]+>", "", str(value or ""))
    return html.unescape(text).strip()


def _parse_through_games_date(html_text):
    match = re.search(r"Through Games\s+([A-Za-z]{3}\.\s+\d{1,2}\s+\d{4})", html_text)
    if not match:
        raise WomensNetParseError("NCAA NET page missing 'Through Games' date.")
    label = match.group(1).replace(".", "")
    return pd.to_datetime(label, format="%b %d %Y").strftime("%Y-%m-%d")


def _extract_rankings_table(html_text):
    match = re.search(r"<table class=\"sticky\">(.*?)</table>", html_text, flags=re.S)
    if not match:
        raise WomensNetParseError("NCAA NET page missing rankings table.")
    return match.group(1)


def fetch_current_net_html(session=None, timeout=20):
    own_session = session is None
    session = session or requests.Session()
    try:
        response = session.get(WOMENS_NET_URL, headers=WOMENS_NET_HEADERS, timeout=timeout)
        response.raise_for_status()
        return response.text
    finally:
        if own_session:
            session.close()


def parse_current_net_html(html_text, source_url=WOMENS_NET_URL, min_teams=0):
    snapshot_date = _parse_through_games_date(html_text)
    table_html = _extract_rankings_table(html_text)
    rows = re.findall(r"<tr>(.*?)</tr>", table_html, flags=re.S)
    if len(rows) < 2:
        raise WomensNetParseError("NCAA NET page did not contain ranking rows.")

    header_cells = re.findall(r"<th[^>]*>(.*?)</th>", rows[0], flags=re.S)
    headers = [_clean_cell(cell) for cell in header_cells]
    expected = ["Rank", "School", "Record", "Conf", "Road", "Neutral", "Home", "Non-Div I", "Prev", "Quad 1", "Quad 2", "Quad 3", "Quad 4"]
    if headers[: len(expected)] != expected:
        raise WomensNetParseError(f"Unexpected NCAA NET headers: {headers[:len(expected)]}")

    parsed_rows = []
    skipped_short_rows = 0
    for row_html in rows[1:]:
        cells = re.findall(r"<td[^>]*>(.*?)</td>", row_html, flags=re.S)
        if len(cells) < len(expected):
            skipped_short_rows += 1
            continue
        values = [_clean_cell(cell) for cell in cells[: len(expected)]]
        parsed_rows.append(
            {
                "snapshot_date": snapshot_date,
                "team": values[1],
                "net_rank": pd.to_numeric(values[0], errors="coerce"),
                "prev_rank": pd.to_numeric(values[8], errors="coerce"),
                "record": values[2],
                "conf": values[3],
                "quad1": values[9],
                "quad2": values[10],
                "quad3": values[11],
                "quad4": values[12],
                "source_url": source_url,
            }
        )

    if not parsed_rows:
        raise WomensNetParseError("Parsed NCAA NET page contained no ranking data.")
    if len(parsed_rows) < int(min_teams or 0):
        raise WomensNetParseError(
            "Parsed NCAA NET page contained too few ranking rows: "
            f"{len(parsed_rows)} parsed, {skipped_short_rows} skipped, expected at least {int(min_teams)}."
        )
    return pd.DataFrame(parsed_rows, columns=WOMENS_NET_SNAPSHOT_COLUMNS)


def load_snapshot_file(path):
    if not path or not os.path.exists(path):
        return _empty_snapshot_frame()
    df = pd.read_csv(path)
    if df.empty:
        return _empty_snapshot_frame()
    for col in WOMENS_NET_SNAPSHOT_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return df[WOMENS_NET_SNAPSHOT_COLUMNS].copy()


def save_snapshot_file(df, path):
    ordered = df.copy()
    for col in WOMENS_NET_SNAPSHOT_COLUMNS:
        if col not in ordered.columns:
            ordered[col] = pd.NA
    ordered = ordered[WOMENS_NET_SNAPSHOT_COLUMNS].sort_values(["snapshot_date", "team"]).reset_index(drop=True)
    ordered.to_csv(path, index=False)


def load_team_map(path, team_names=None):
    if path and os.path.exists(path):
        df = pd.read_csv(path)
        for col in WOMENS_NET_MAP_COLUMNS:
            if col not in df.columns:
                df[col] = pd.NA
        return df[WOMENS_NET_MAP_COLUMNS].copy()
    return _empty_match_frame(list(team_names) if team_names is not None else [])


def save_team_map(df, path):
    ordered = df.copy()
    for col in WOMENS_NET_MAP_COLUMNS:
        if col not in ordered.columns:
            ordered[col] = pd.NA
    ordered = ordered[WOMENS_NET_MAP_COLUMNS].sort_values("team").reset_index(drop=True)
    ordered.to_csv(path, index=False)


def build_team_map(team_names, net_team_names, existing_map=None):
    existing_map = existing_map if existing_map is not None else _empty_match_frame([])
    existing_lookup = existing_map.set_index("team").to_dict("index") if not existing_map.empty else {}
    normalized_lookup = {}
    for net_team in sorted(set(str(name) for name in net_team_names if str(name).strip())):
        normalized_lookup.setdefault(_normalize_name(net_team), []).append(net_team)

    rows = []
    for team in sorted(set(str(name) for name in team_names if str(name).strip())):
        existing_row = existing_lookup.get(team)
        if (
            existing_row
            and pd.notna(existing_row.get("net_team"))
            and str(existing_row.get("match_source", "")).lower() == "manual"
        ):
            rows.append(
                {
                    "team": team,
                    "net_team": existing_row["net_team"],
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

        if matched is None:
            for key in candidate_keys:
                if len(key.split()) < 2:
                    continue
                contains = [
                    candidate
                    for candidate in normalized_lookup
                    if candidate.startswith(f"{key} ")
                    or candidate.endswith(f" {key}")
                    or f" {key} " in candidate
                ]
                if len(contains) != 1:
                    continue
                resolved = normalized_lookup[contains[0]]
                if len(resolved) != 1:
                    continue
                score = SequenceMatcher(None, key, contains[0]).ratio()
                matched = (
                    resolved[0],
                    key,
                    score,
                    "contains" if key == _normalize_name(team) else "alias_contains",
                )
                break

        if matched is not None:
            rows.append(
                {
                    "team": team,
                    "net_team": matched[0],
                    "match_source": matched[3],
                    "match_score": round(float(matched[2]), 4),
                    "needs_review": False,
                }
            )
        else:
            rows.append(
                {
                    "team": team,
                    "net_team": pd.NA,
                    "match_source": "unmatched",
                    "match_score": 0.0,
                    "needs_review": True,
                }
            )

    return pd.DataFrame(rows, columns=WOMENS_NET_MAP_COLUMNS)


def sync_team_map_for_games(games_df, snapshots_df, team_map_df=None):
    all_teams = pd.concat([games_df["team"], games_df["opponent"]], ignore_index=True).dropna().unique()
    snapshot_teams = snapshots_df["team"].dropna().unique() if not snapshots_df.empty else []
    return build_team_map(all_teams, snapshot_teams, existing_map=team_map_df)


def ensure_womens_net_feature_columns(df):
    for col in WOMENS_NET_GAME_FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0
    return df


def _merge_side_snapshot(df, snapshots_df, left_team_col, prefix):
    left = df.copy()
    left[left_team_col] = left[left_team_col].fillna("")
    left["lookup_date"] = pd.to_datetime(left["date"], errors="coerce") - pd.Timedelta(days=1)
    left = left.sort_values(["lookup_date", left_team_col]).reset_index(drop=True)

    right = snapshots_df.copy()
    right["snapshot_date"] = pd.to_datetime(right["snapshot_date"], errors="coerce")
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
        "net_rank": f"{prefix}net_rank",
        "prev_rank": f"{prefix}prev_rank",
    }
    merged = merged.rename(columns=rename_map)
    drop_cols = [col for col in ["snapshot_date", "lookup_date", "record", "conf", "quad1", "quad2", "quad3", "quad4", "source_url"] if col in merged.columns]
    return merged.drop(columns=drop_cols)


def _rank_to_strength(series):
    numeric = pd.to_numeric(series, errors="coerce")
    max_rank = numeric.max(skipna=True)
    scale_max = max(float(max_rank) if pd.notna(max_rank) else MIN_WOMENS_NET_TEAMS, MIN_WOMENS_NET_TEAMS)
    clipped = numeric.clip(lower=1.0, upper=scale_max)
    return 1.0 - ((clipped - 1.0) / max(scale_max - 1.0, 1.0))


def add_womens_net_features(df, snapshots_df, team_map_df):
    df = ensure_womens_net_feature_columns(df.copy())
    if df.empty or snapshots_df.empty or team_map_df.empty:
        return df

    mapped = df.drop(columns=[col for col in WOMENS_NET_GAME_FEATURE_COLUMNS if col in df.columns]).copy()
    lookup = team_map_df[["team", "net_team"]].drop_duplicates()
    mapped = mapped.merge(lookup.rename(columns={"team": "team", "net_team": "team_net"}), on="team", how="left")
    mapped = mapped.merge(lookup.rename(columns={"team": "opponent", "net_team": "opponent_net"}), on="opponent", how="left")

    merged = _merge_side_snapshot(mapped, snapshots_df, "team_net", "team_")
    merged = _merge_side_snapshot(merged, snapshots_df, "opponent_net", "opp_")

    merged["womens_net_diff_rank_strength"] = _rank_to_strength(merged.get("team_net_rank")) - _rank_to_strength(
        merged.get("opp_net_rank")
    )
    merged["womens_net_diff_prev_rank_strength"] = _rank_to_strength(merged.get("team_prev_rank")) - _rank_to_strength(
        merged.get("opp_prev_rank")
    )

    for col in WOMENS_NET_GAME_FEATURE_COLUMNS:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)

    drop_cols = [
        "team_net",
        "opponent_net",
        "team_net_rank",
        "team_prev_rank",
        "opp_net_rank",
        "opp_prev_rank",
    ]
    merged = merged.drop(columns=[col for col in drop_cols if col in merged.columns])
    return ensure_womens_net_feature_columns(merged)


def matchup_features_for_game(home_team, away_team, game_date, snapshots_df, team_map_df):
    frame = pd.DataFrame([{"team": home_team, "opponent": away_team, "date": pd.Timestamp(game_date).strftime("%Y-%m-%d")}])
    enriched = add_womens_net_features(frame, snapshots_df, team_map_df)
    if enriched.empty:
        return {col: 0.0 for col in WOMENS_NET_GAME_FEATURE_COLUMNS}
    row = enriched.iloc[0]
    return {col: float(pd.to_numeric(row.get(col), errors="coerce") or 0.0) for col in WOMENS_NET_GAME_FEATURE_COLUMNS}


def sync_current_snapshot(league="womens", timeout=20):
    league = normalize_league(league)
    if league != "womens":
        raise ValueError("NCAA women NET sync is only supported for the women's pipeline.")

    paths = get_league_artifact_paths(BASE_DIR, league)
    data_file = paths["data_file"]
    snapshot_file = paths["womens_net_snapshot_file"]
    map_file = paths["womens_net_map_file"]
    if not data_file or not os.path.exists(data_file):
        raise FileNotFoundError(f"Processed data file not found: {data_file}")

    html_text = fetch_current_net_html(timeout=timeout)
    fetched = parse_current_net_html(html_text, min_teams=MIN_WOMENS_NET_TEAMS)
    existing = load_snapshot_file(snapshot_file)
    combined = pd.concat([existing, fetched], ignore_index=True)
    combined = combined.drop_duplicates(subset=["snapshot_date", "team"], keep="last")

    games_df = pd.read_csv(data_file, low_memory=False)
    team_map = load_team_map(map_file, pd.concat([games_df["team"], games_df["opponent"]]).dropna().unique())
    team_map = sync_team_map_for_games(games_df, combined, team_map_df=team_map)
    save_snapshot_file(combined, snapshot_file)
    save_team_map(team_map, map_file)

    mapped_teams = team_map.loc[team_map["net_team"].notna(), "team"].nunique() if not team_map.empty else 0
    matched_net_teams = team_map.loc[team_map["net_team"].notna(), "net_team"].nunique() if not team_map.empty else 0
    all_teams = pd.concat([games_df["team"], games_df["opponent"]], ignore_index=True).dropna().nunique()
    return {
        "games": int(len(games_df)),
        "team_map_coverage": (mapped_teams / all_teams) if all_teams else 0.0,
        "source_team_coverage": (matched_net_teams / fetched["team"].nunique()) if not fetched.empty else 0.0,
        "snapshot_dates": int(pd.to_datetime(combined["snapshot_date"], errors="coerce").dropna().nunique()),
    }


def main():
    parser = argparse.ArgumentParser(description="Manage NCAA women's NET snapshots.")
    parser.add_argument("command", choices=["sync"], help="sync current snapshot")
    parser.add_argument("--league", default="womens", help="League to operate on")
    parser.add_argument("--timeout", type=int, default=20, help="HTTP timeout in seconds")
    args = parser.parse_args()

    report = sync_current_snapshot(league=args.league, timeout=args.timeout)
    print(f"games={report['games']}")
    print(f"team_map_coverage={report['team_map_coverage']:.1%}")
    print(f"source_team_coverage={report['source_team_coverage']:.1%}")
    print(f"snapshot_dates={report['snapshot_dates']}")


if __name__ == "__main__":
    main()
