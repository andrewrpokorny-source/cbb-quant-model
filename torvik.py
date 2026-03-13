import argparse
import os
import re
import json
from dataclasses import dataclass
from datetime import timedelta
from difflib import SequenceMatcher, get_close_matches
from html.parser import HTMLParser
from io import StringIO
from typing import Iterable
from urllib.parse import urlencode

import pandas as pd
import requests

from league_config import get_league_artifact_paths, get_season_start_date, normalize_league


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TORVIK_BASE_URL = "https://barttorvik.com/trank.php"
TORVIK_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://barttorvik.com/",
}
TORVIK_SNAPSHOT_COLUMNS = [
    "snapshot_date",
    "season",
    "team",
    "conf",
    "games",
    "record",
    "adj_oe",
    "adj_de",
    "barthag",
    "adj_tempo",
    "efg_off",
    "efg_def",
    "tor_off",
    "tor_def",
    "orb_off",
    "drb_def",
    "ftr_off",
    "ftr_def",
    "wab",
    "source_url",
]
TORVIK_METRIC_COLUMNS = [
    "adj_oe",
    "adj_de",
    "barthag",
    "adj_tempo",
    "efg_off",
    "efg_def",
    "tor_off",
    "tor_def",
    "orb_off",
    "drb_def",
    "ftr_off",
    "ftr_def",
    "wab",
]
TORVIK_GAME_FEATURE_COLUMNS = [
    "torvik_team_adj_oe",
    "torvik_team_adj_de",
    "torvik_team_barthag",
    "torvik_team_adj_tempo",
    "torvik_opp_adj_oe",
    "torvik_opp_adj_de",
    "torvik_opp_barthag",
    "torvik_opp_adj_tempo",
    "torvik_diff_adj_oe",
    "torvik_diff_adj_de",
    "torvik_diff_barthag",
    "torvik_tempo_gap",
    "torvik_diff_efg",
    "torvik_diff_tor",
    "torvik_diff_orb",
    "torvik_diff_ftr",
]
TORVIK_REQUIRED_COLUMNS = ["team", "opponent", "date"]
TORVIK_MATCH_COLUMNS = ["team", "torvik_team", "match_source", "match_score", "needs_review"]
TORVIK_NAME_OVERRIDES = {
    "california baptist": "cal baptist",
    "college of charleston": "charleston",
    "boston college eagles": "boston college",
    "boston university terriers": "boston university",
    "grambling tigers": "grambling st",
    "iu indianapolis jaguars": "iu indy",
    "uic": "illinois chicago",
    "long island university sharks": "liu",
    "loyola maryland greyhounds": "loyola md",
    "mass lowell": "umass lowell",
    "maryland eastern shore": "maryland eastern shore",
    "mcneese cowboys": "mcneese st",
    "mississippi": "mississippi",
    "mount st marys": "mount st marys",
    "nc state": "nc state",
    "nicholls colonels": "nicholls st",
    "omaha mavericks": "nebraska omaha",
    "north carolina state": "nc state",
    "ole miss": "mississippi",
    "saint josephs": "st josephs",
    "saint marys": "st marys",
    "saint peters": "st peters",
    "sam houston bearkats": "sam houston st",
    "se louisiana lions": "southeastern louisiana",
    "southern methodist": "smu",
    "st thomas minnesota tommies": "st thomas",
    "texas am corpus christi": "texas am corpus chris",
    "ul monroe warhawks": "louisiana monroe",
    "ut martin skyhawks": "tennessee martin",
    "ut rio grande valley": "ut rio grande valley",
}
TEAM_MAP_FILE = os.path.join(BASE_DIR, "team_map.json")
_EXTERNAL_TEAM_MAP = None


class TorvikParseError(ValueError):
    """Raised when a Torvik table cannot be parsed into the expected schema."""


class _TorvikTableParser(HTMLParser):
    """Minimal HTML table extractor for Torvik's ratings page."""

    def __init__(self):
        super().__init__()
        self._table_depth = 0
        self._in_caption = False
        self._target_table = False
        self._collecting_target = False
        self._current_row = None
        self._current_cell = None
        self.rows = []
        self.caption_text = ""

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self._table_depth += 1
            self._target_table = False
            self.caption_text = ""
        elif tag == "caption" and self._table_depth:
            self._in_caption = True
            self.caption_text = ""
        elif tag == "tr" and self._collecting_target:
            self._current_row = []
        elif tag in {"th", "td"} and self._current_row is not None:
            self._current_cell = []

    def handle_data(self, data):
        if self._in_caption:
            self.caption_text += data
        if self._current_cell is not None:
            self._current_cell.append(data)

    def handle_endtag(self, tag):
        if tag == "caption" and self._table_depth:
            self._in_caption = False
            if "T-Rank and Tempo-Free Stats" in " ".join(self.caption_text.split()):
                self._target_table = True
                self._collecting_target = True
        elif tag in {"th", "td"} and self._current_cell is not None and self._current_row is not None:
            text = " ".join("".join(self._current_cell).split())
            self._current_row.append(text)
            self._current_cell = None
        elif tag == "tr" and self._current_row is not None:
            if any(cell for cell in self._current_row):
                self.rows.append(self._current_row)
            self._current_row = None
        elif tag == "table" and self._table_depth:
            if self._target_table:
                self._collecting_target = False
            self._table_depth -= 1
            self._target_table = False


@dataclass(frozen=True)
class TorvikSnapshotRequest:
    season: int
    begin: pd.Timestamp
    end: pd.Timestamp

    @property
    def params(self):
        return {
            "year": int(self.season),
            "begin": self.begin.strftime("%Y%m%d"),
            "end": self.end.strftime("%Y%m%d"),
            "top": 0,
            "conlimit": "All",
            "state": "All",
        }

    @property
    def url(self):
        return f"{TORVIK_BASE_URL}?{urlencode(self.params)}"


def _coerce_ts(value):
    return pd.Timestamp(value).normalize()


def _season_year_for_date(value):
    ts = _coerce_ts(value)
    return ts.year + 1 if ts.month >= 11 else ts.year


def _season_start_for_date(value):
    ts = _coerce_ts(value)
    start_year = ts.year if ts.month >= 11 else ts.year - 1
    return pd.Timestamp(year=start_year, month=11, day=1)


def _request_for_snapshot_date(snapshot_date):
    ts = _coerce_ts(snapshot_date)
    return TorvikSnapshotRequest(
        season=_season_year_for_date(ts),
        begin=_season_start_for_date(ts),
        end=ts,
    )


def _looks_like_browser_verification(html):
    lowered = str(html or "").lower()
    return "verifying browser" in lowered and "js_test_submitted" in lowered


def _normalize_name(value):
    text = str(value or "").strip().lower()
    if not text:
        return ""
    replacements = {
        "&": " and ",
        "@": " at ",
        "'": "",
        ".": " ",
        "-": " ",
        "/": " ",
    }
    for src, dest in replacements.items():
        text = text.replace(src, dest)
    text = re.sub(r"\bsaint\b", "st", text)
    text = re.sub(r"\bmount\b", "mt", text)
    text = re.sub(r"\bstate\b", "st", text)
    text = re.sub(r"\s+", " ", text).strip()
    return TORVIK_NAME_OVERRIDES.get(text, text)


def _extract_first_number(value):
    match = re.search(r"[-+]?\d*\.?\d+", str(value or ""))
    return float(match.group()) if match else pd.NA


def _clean_team_cell(value):
    text = " ".join(str(value or "").split())
    text = re.split(r"\s(?:vs\.|@|at)\s", text, maxsplit=1)[0]
    return text.strip()


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


def _candidate_match_keys(team):
    text = str(team or "").strip()
    if not text:
        return []

    candidates = [text]
    external_map = _load_external_team_map()
    mapped = external_map.get(text)
    if mapped:
        candidates.append(mapped)

    tokens = text.split()
    for trim in (1, 2):
        if len(tokens) > trim:
            candidates.append(" ".join(tokens[:-trim]))

    normalized_candidates = []
    for candidate in candidates:
        normalized = _normalize_name(candidate)
        if normalized and normalized not in normalized_candidates:
            normalized_candidates.append(normalized)
    return normalized_candidates


def _empty_snapshot_frame():
    return pd.DataFrame(columns=TORVIK_SNAPSHOT_COLUMNS)


def _empty_match_frame(team_names: Iterable[str]):
    return pd.DataFrame(
        [
            {
                "team": team,
                "torvik_team": pd.NA,
                "match_source": "unmatched",
                "match_score": 0.0,
                "needs_review": True,
            }
            for team in sorted(set(team_names))
        ],
        columns=TORVIK_MATCH_COLUMNS,
    )


def ensure_torvik_feature_columns(df):
    for col in TORVIK_GAME_FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0
    return df


def _parse_torvik_rows(html):
    parser = _TorvikTableParser()
    parser.feed(html)
    if not parser.rows:
        raise TorvikParseError("Torvik ratings table was not found in the HTML response.")
    return parser.rows


def parse_ratings_html(html, snapshot_date, source_url):
    rows = _parse_torvik_rows(html)
    header_idx = None
    for idx, row in enumerate(rows):
        row_set = set(row)
        if {"Team", "AdjOE", "AdjDE", "Barthag"}.issubset(row_set):
            header_idx = idx
            break

    if header_idx is None:
        raise TorvikParseError("Torvik ratings header row was not found.")

    header = rows[header_idx]
    records = []
    expected_len = len(header)

    for row in rows[header_idx + 1 :]:
        if row == header or not row:
            continue
        if row[0] == "Rk" or len(row) < expected_len:
            continue
        try:
            rank = int(str(row[0]))
        except ValueError:
            continue

        record = {
            "snapshot_date": _coerce_ts(snapshot_date).strftime("%Y-%m-%d"),
            "season": _season_year_for_date(snapshot_date),
            "team": _clean_team_cell(row[1]),
            "conf": row[2],
            "games": int(_extract_first_number(row[3])),
            "record": row[4],
            "adj_oe": _extract_first_number(row[5]),
            "adj_de": _extract_first_number(row[6]),
            "barthag": _extract_first_number(row[7]),
            "efg_off": _extract_first_number(row[8]),
            "efg_def": _extract_first_number(row[9]),
            "tor_off": _extract_first_number(row[10]),
            "tor_def": _extract_first_number(row[11]),
            "orb_off": _extract_first_number(row[12]),
            "drb_def": _extract_first_number(row[13]),
            "ftr_off": _extract_first_number(row[14]),
            "ftr_def": _extract_first_number(row[15]),
            "adj_tempo": _extract_first_number(row[22]),
            "wab": _extract_first_number(row[23]),
            "source_url": source_url,
        }
        if rank > 0 and record["team"]:
            records.append(record)

    if not records:
        raise TorvikParseError("Torvik ratings rows were parsed but no team records were extracted.")

    return pd.DataFrame(records, columns=TORVIK_SNAPSHOT_COLUMNS)


def fetch_ratings_snapshot(snapshot_date, session=None, timeout=20):
    request = _request_for_snapshot_date(snapshot_date)
    own_session = session is None
    session = session or requests.Session()
    session.headers.update(TORVIK_HEADERS)

    try:
        response = session.get(request.url, timeout=timeout)
        response.raise_for_status()
        html = response.text
        if _looks_like_browser_verification(html):
            response = session.post(
                request.url,
                data={"js_test_submitted": "1"},
                timeout=timeout,
            )
            response.raise_for_status()
            html = response.text
        if _looks_like_browser_verification(html):
            raise TorvikParseError("Torvik browser verification challenge could not be bypassed.")
        return parse_ratings_html(html, snapshot_date=request.end, source_url=request.url)
    finally:
        if own_session:
            session.close()


def load_snapshot_file(path):
    if not path or not os.path.exists(path):
        return _empty_snapshot_frame()
    df = pd.read_csv(path)
    if df.empty:
        return _empty_snapshot_frame()
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"]).dt.strftime("%Y-%m-%d")
    return df[sorted(set(df.columns) | set(TORVIK_SNAPSHOT_COLUMNS))].copy()


def save_snapshot_file(df, path):
    ordered = df.copy()
    for col in TORVIK_SNAPSHOT_COLUMNS:
        if col not in ordered.columns:
            ordered[col] = pd.NA
    ordered = ordered[TORVIK_SNAPSHOT_COLUMNS].sort_values(["snapshot_date", "team"]).reset_index(drop=True)
    ordered.to_csv(path, index=False)


def load_team_map(path, team_names=None):
    if path and os.path.exists(path):
        df = pd.read_csv(path)
        for col in TORVIK_MATCH_COLUMNS:
            if col not in df.columns:
                df[col] = pd.NA
        return df[TORVIK_MATCH_COLUMNS].copy()
    team_names = [] if team_names is None else list(team_names)
    return _empty_match_frame(team_names)


def save_team_map(df, path):
    ordered = df.copy()
    for col in TORVIK_MATCH_COLUMNS:
        if col not in ordered.columns:
            ordered[col] = pd.NA
    ordered = ordered[TORVIK_MATCH_COLUMNS].sort_values("team").reset_index(drop=True)
    ordered.to_csv(path, index=False)


def build_team_map(team_names, torvik_team_names, existing_map=None):
    existing_map = existing_map if existing_map is not None else _empty_match_frame([])
    existing_lookup = existing_map.set_index("team").to_dict("index") if not existing_map.empty else {}

    torvik_names = sorted(set(str(name) for name in torvik_team_names if str(name).strip()))
    normalized_lookup = {}
    for torvik_team in torvik_names:
        normalized_lookup.setdefault(_normalize_name(torvik_team), []).append(torvik_team)

    rows = []
    for team in sorted(set(str(name) for name in team_names if str(name).strip())):
        existing_row = existing_lookup.get(team)
        if existing_row and pd.notna(existing_row.get("torvik_team")):
            rows.append(
                {
                    "team": team,
                    "torvik_team": existing_row["torvik_team"],
                    "match_source": existing_row.get("match_source", "manual"),
                    "match_score": float(existing_row.get("match_score", 1.0) or 1.0),
                    "needs_review": bool(existing_row.get("needs_review", False)),
                }
            )
            continue

        match_keys = _candidate_match_keys(team)
        exact_match = None
        for match_key in match_keys:
            matches = normalized_lookup.get(match_key, [])
            if len(matches) == 1:
                exact_match = (matches[0], match_key)
                break
        if exact_match is not None:
            torvik_team, match_key = exact_match
            rows.append(
                {
                    "team": team,
                    "torvik_team": torvik_team,
                    "match_source": "normalized_exact" if match_key == _normalize_name(team) else "alias_exact",
                    "match_score": 1.0,
                    "needs_review": False,
                }
            )
            continue

        best_match = None
        for match_key in match_keys:
            close = get_close_matches(match_key, list(normalized_lookup), n=2, cutoff=0.82)
            if not close:
                continue
            best_key = close[0]
            best_score = SequenceMatcher(None, match_key, best_key).ratio()
            second_score = SequenceMatcher(None, match_key, close[1]).ratio() if len(close) > 1 else 0.0
            candidates = normalized_lookup.get(best_key, [])
            if len(candidates) != 1:
                continue
            if best_match is None or best_score > best_match[2]:
                best_match = (candidates[0], match_key, best_score, second_score)

        if best_match is not None:
            torvik_team, match_key, best_score, second_score = best_match
            if best_score >= 0.9 and (best_score - second_score) >= 0.03:
                rows.append(
                    {
                        "team": team,
                        "torvik_team": torvik_team,
                        "match_source": "fuzzy" if match_key == _normalize_name(team) else "alias_fuzzy",
                        "match_score": round(best_score, 4),
                        "needs_review": False,
                    }
                )
                continue

        rows.append(
            {
                "team": team,
                "torvik_team": pd.NA,
                "match_source": "unmatched",
                "match_score": 0.0,
                "needs_review": True,
            }
        )

    return pd.DataFrame(rows, columns=TORVIK_MATCH_COLUMNS)


def sync_team_map_for_games(games_df, snapshots_df, team_map_df=None):
    if games_df.empty or snapshots_df.empty:
        return load_team_map(None, games_df.get("team", pd.Series(dtype=str)).tolist())

    all_teams = pd.concat([games_df["team"], games_df["opponent"]], ignore_index=True).dropna().unique()
    snapshot_teams = snapshots_df["team"].dropna().unique()
    return build_team_map(all_teams, snapshot_teams, existing_map=team_map_df)


def _merge_side_snapshot(df, snapshots_df, left_team_col, prefix):
    if snapshots_df.empty:
        for col in TORVIK_METRIC_COLUMNS:
            df[f"{prefix}{col}"] = 0.0
        return df

    left = df.copy()
    left["lookup_date"] = pd.to_datetime(left["date"]) - timedelta(days=1)
    left[left_team_col] = left[left_team_col].fillna("")
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

    rename_map = {col: f"{prefix}{col}" for col in TORVIK_METRIC_COLUMNS}
    merged = merged.rename(columns=rename_map)
    drop_cols = [col for col in ["snapshot_date", "season", "conf", "games", "record", "source_url"] if col in merged]
    merged = merged.drop(columns=drop_cols)
    return merged


def add_torvik_features(df, snapshots_df, team_map_df):
    df = ensure_torvik_feature_columns(df.copy())
    if df.empty or snapshots_df.empty or team_map_df.empty:
        return df

    for required in TORVIK_REQUIRED_COLUMNS:
        if required not in df.columns:
            return df

    mapped = df.drop(columns=[col for col in TORVIK_GAME_FEATURE_COLUMNS if col in df.columns]).copy()
    map_lookup = team_map_df[["team", "torvik_team"]].drop_duplicates()
    mapped = mapped.merge(map_lookup.rename(columns={"team": "team", "torvik_team": "team_torvik"}), on="team", how="left")
    mapped = mapped.merge(
        map_lookup.rename(columns={"team": "opponent", "torvik_team": "opponent_torvik"}),
        on="opponent",
        how="left",
    )

    snapshots = snapshots_df.copy()
    snapshots["snapshot_date"] = pd.to_datetime(snapshots["snapshot_date"]).dt.normalize()
    snapshots = snapshots.sort_values(["team", "snapshot_date"]).reset_index(drop=True)

    merged = _merge_side_snapshot(mapped, snapshots, "team_torvik", "torvik_team_")
    merged = _merge_side_snapshot(merged, snapshots, "opponent_torvik", "torvik_opp_")

    metric_defaults = {
        "torvik_team_adj_oe": 0.0,
        "torvik_team_adj_de": 0.0,
        "torvik_team_barthag": 0.0,
        "torvik_team_adj_tempo": 0.0,
        "torvik_opp_adj_oe": 0.0,
        "torvik_opp_adj_de": 0.0,
        "torvik_opp_barthag": 0.0,
        "torvik_opp_adj_tempo": 0.0,
    }
    for col, default in metric_defaults.items():
        merged[col] = pd.to_numeric(merged.get(col), errors="coerce").fillna(default)

    merged["torvik_diff_adj_oe"] = merged["torvik_team_adj_oe"] - merged["torvik_opp_adj_oe"]
    merged["torvik_diff_adj_de"] = merged["torvik_opp_adj_de"] - merged["torvik_team_adj_de"]
    merged["torvik_diff_barthag"] = merged["torvik_team_barthag"] - merged["torvik_opp_barthag"]
    merged["torvik_tempo_gap"] = merged["torvik_team_adj_tempo"] - merged["torvik_opp_adj_tempo"]
    merged["torvik_diff_efg"] = (
        pd.to_numeric(merged.get("torvik_team_efg_off"), errors="coerce").fillna(0.0)
        - pd.to_numeric(merged.get("torvik_opp_efg_off"), errors="coerce").fillna(0.0)
    )
    merged["torvik_diff_tor"] = (
        pd.to_numeric(merged.get("torvik_opp_tor_off"), errors="coerce").fillna(0.0)
        - pd.to_numeric(merged.get("torvik_team_tor_off"), errors="coerce").fillna(0.0)
    )
    merged["torvik_diff_orb"] = (
        pd.to_numeric(merged.get("torvik_team_orb_off"), errors="coerce").fillna(0.0)
        - pd.to_numeric(merged.get("torvik_opp_orb_off"), errors="coerce").fillna(0.0)
    )
    merged["torvik_diff_ftr"] = (
        pd.to_numeric(merged.get("torvik_team_ftr_off"), errors="coerce").fillna(0.0)
        - pd.to_numeric(merged.get("torvik_opp_ftr_off"), errors="coerce").fillna(0.0)
    )

    drop_cols = [
        "team_torvik",
        "opponent_torvik",
        "lookup_date",
        "torvik_team_efg_off",
        "torvik_team_efg_def",
        "torvik_team_tor_off",
        "torvik_team_tor_def",
        "torvik_team_orb_off",
        "torvik_team_drb_def",
        "torvik_team_ftr_off",
        "torvik_team_ftr_def",
        "torvik_team_wab",
        "torvik_opp_efg_off",
        "torvik_opp_efg_def",
        "torvik_opp_tor_off",
        "torvik_opp_tor_def",
        "torvik_opp_orb_off",
        "torvik_opp_drb_def",
        "torvik_opp_ftr_off",
        "torvik_opp_ftr_def",
        "torvik_opp_wab",
    ]
    merged = merged.drop(columns=[col for col in drop_cols if col in merged.columns])
    return ensure_torvik_feature_columns(merged)


def matchup_features_for_game(home_team, away_team, game_date, snapshots_df, team_map_df):
    frame = pd.DataFrame(
        [{"team": home_team, "opponent": away_team, "date": pd.Timestamp(game_date).strftime("%Y-%m-%d")}]
    )
    enriched = add_torvik_features(frame, snapshots_df, team_map_df)
    if enriched.empty:
        return {col: 0.0 for col in TORVIK_GAME_FEATURE_COLUMNS}
    row = enriched.iloc[0]
    return {col: float(pd.to_numeric(row.get(col), errors="coerce") or 0.0) for col in TORVIK_GAME_FEATURE_COLUMNS}


def build_missing_snapshot_dates(games_df):
    if games_df.empty or "date" not in games_df.columns:
        return []
    dates = pd.to_datetime(games_df["date"], errors="coerce").dropna().dt.normalize().unique()
    return sorted({(pd.Timestamp(date) - timedelta(days=1)).normalize() for date in dates})


def update_snapshot_file_for_games(games_df, snapshot_path, timeout=20):
    existing = load_snapshot_file(snapshot_path)
    have_dates = set(pd.to_datetime(existing["snapshot_date"], errors="coerce").dropna().dt.normalize()) if not existing.empty else set()
    target_dates = [date for date in build_missing_snapshot_dates(games_df) if date not in have_dates]

    if not target_dates:
        return existing

    session = requests.Session()
    session.headers.update(TORVIK_HEADERS)
    fetched_frames = []
    try:
        for idx, snapshot_date in enumerate(target_dates, start=1):
            print(f"   -> Torvik snapshot {idx}/{len(target_dates)}: {snapshot_date.date()}")
            fetched_frames.append(fetch_ratings_snapshot(snapshot_date, session=session, timeout=timeout))
    finally:
        session.close()

    combined = pd.concat([existing, *fetched_frames], ignore_index=True) if fetched_frames else existing
    combined["snapshot_date"] = pd.to_datetime(combined["snapshot_date"]).dt.strftime("%Y-%m-%d")
    combined = combined.drop_duplicates(subset=["snapshot_date", "team"], keep="last")
    save_snapshot_file(combined, snapshot_path)
    return combined


def coverage_report(games_df, snapshots_df, team_map_df):
    if games_df.empty:
        return {"games": 0, "team_map_coverage": 0.0, "snapshot_dates": 0}

    mapped_teams = (
        team_map_df.loc[team_map_df["torvik_team"].notna(), "team"].nunique() if not team_map_df.empty else 0
    )
    all_teams = pd.concat([games_df["team"], games_df["opponent"]], ignore_index=True).dropna().nunique()
    return {
        "games": int(len(games_df)),
        "team_map_coverage": (mapped_teams / all_teams) if all_teams else 0.0,
        "snapshot_dates": int(pd.to_datetime(snapshots_df["snapshot_date"], errors="coerce").dropna().nunique())
        if not snapshots_df.empty
        else 0,
    }


def sync_from_processed_data(league="mens", timeout=20):
    league = normalize_league(league)
    paths = get_league_artifact_paths(BASE_DIR, league)
    data_file = paths["data_file"]
    snapshot_file = paths["torvik_snapshot_file"]
    map_file = paths["torvik_map_file"]

    if league != "mens":
        raise ValueError("Torvik sync is currently supported for the men's pipeline only.")
    if not data_file or not os.path.exists(data_file):
        raise FileNotFoundError(f"Processed data file not found: {data_file}")

    games_df = pd.read_csv(data_file, low_memory=False)
    if "date" in games_df.columns:
        season_start = pd.Timestamp(get_season_start_date(league))
        game_dates = pd.to_datetime(games_df["date"], errors="coerce")
        games_df = games_df.loc[game_dates >= season_start].copy()
    snapshots_df = update_snapshot_file_for_games(games_df, snapshot_file, timeout=timeout)
    team_map_df = load_team_map(map_file, pd.concat([games_df["team"], games_df["opponent"]]).dropna().unique())
    team_map_df = sync_team_map_for_games(games_df, snapshots_df, team_map_df=team_map_df)
    save_team_map(team_map_df, map_file)
    return coverage_report(games_df, snapshots_df, team_map_df)


def main():
    parser = argparse.ArgumentParser(description="Manage historical Bart Torvik ratings snapshots.")
    parser.add_argument(
        "command",
        choices=["sync", "coverage"],
        help="sync: fetch missing snapshots and refresh the team map; coverage: report snapshot/map status.",
    )
    parser.add_argument("--league", default="mens", help="League to operate on (currently mens only).")
    parser.add_argument("--timeout", type=int, default=20, help="HTTP timeout in seconds.")
    args = parser.parse_args()

    league = normalize_league(args.league)
    paths = get_league_artifact_paths(BASE_DIR, league)
    data_file = paths["data_file"]
    snapshot_file = paths["torvik_snapshot_file"]
    map_file = paths["torvik_map_file"]

    if args.command == "sync":
        report = sync_from_processed_data(league=league, timeout=args.timeout)
    else:
        games_df = pd.read_csv(data_file, low_memory=False) if data_file and os.path.exists(data_file) else pd.DataFrame()
        snapshots_df = load_snapshot_file(snapshot_file)
        team_map_df = load_team_map(map_file, pd.concat([games_df.get("team", pd.Series(dtype=str)), games_df.get("opponent", pd.Series(dtype=str))]).dropna().unique())
        report = coverage_report(games_df, snapshots_df, team_map_df)

    print(f"games={report['games']}")
    print(f"team_map_coverage={report['team_map_coverage']:.1%}")
    print(f"snapshot_dates={report['snapshot_dates']}")


if __name__ == "__main__":
    main()
