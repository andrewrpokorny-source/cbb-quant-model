# MLB Market Anchor V2

Track 2 is blocked on source data, not model code. The current
`data/mlb_training_data_processed.csv` has the `moneyline`, `run_line`, and
`total_line` columns, but moneyline coverage is 0%, so the existing frozen
anchor cannot answer market-aware research questions.

Use the diagnostic command before freezing any market-aware anchor:

```bash
uv run python mlb_research/anchor_v2/build_market_anchor.py --diagnose-only
```

The freeze command intentionally fails unless paired moneyline coverage clears
the configured threshold:

```bash
uv run python mlb_research/anchor_v2/build_market_anchor.py --min-coverage 0.95
```

When historical odds are backfilled, this builder adds:

- `team_moneyline`
- `opp_moneyline`
- `market_team_implied_prob`
- `market_opp_implied_prob`
- `market_overround`
- `market_team_no_vig_prob`
- `market_home_no_vig_prob`

The next real research run should use `market_home_no_vig_prob` as a feature
and `roi_mode="moneyline"` in configs so ROI is scored at actual prices rather
than flat -110.

## Odds Source Requirement

Yes, Track 2 needs an odds source before it can produce model conclusions.
The minimum acceptable source is historical game-level MLB moneylines for both
sides of each game, keyed closely enough to join on date, teams, and preferably
game time for doubleheaders. Closing lines are preferred; a fixed pre-game
snapshot time is acceptable if it is consistent and documented.

The builder is intentionally vendor-neutral. It expects those odds to land in
`data/mlb_training_data_processed.csv` as paired `moneyline` values before
freezing a market anchor.
